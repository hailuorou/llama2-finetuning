import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from torch.cuda.amp import autocast as autocast
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

# from vllm.model_executor.models.chatglm import ChatGLMModel 


from .modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMModel

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip

import sys
sys.path.append('/zeng_chao/code/JiuTian/models')
from eva_vit_spatial import VisionTransformer_spatial

__all__ = ["run_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, ChatGLMModel):
        layers = model.encoder.layers
    elif isinstance(model, VisionTransformer_spatial):
        layers = model.blocks
    elif isinstance(model, GPTNeoXForCausalLM):
        layers = model.gpt_neox.layers
    elif isinstance(model, T5ForConditionalGeneration):
        layers = model.encoder.block
    elif isinstance(model, ChatGLMForConditionalGeneration):
        layers = model.transformer.encoder.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers

def get_blocks2(model):
    if isinstance(model, T5ForConditionalGeneration):
        layers = model.decoder.block
    else:
        raise NotImplementedError(type(model))
    return layers
    
def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, VisionTransformer_spatial):
        model.patch_embed = model.patch_embed.to(device)
    elif isinstance(model, ChatGLMModel):
        model.embedding = model.embedding.to(device)
    elif isinstance(model, GPTNeoXForCausalLM):
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, T5ForConditionalGeneration):
        # model.encoder.embed_tokens = model.encoder.embed_tokens.to(device)
        model.shared = model.shared.to(device)
    elif isinstance(model, ChatGLMForConditionalGeneration):
        model.transformer.embedding = model.transformer.embedding.to(device)
        model.transformer.rotary_pos_emb = model.transformer.rotary_pos_emb.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    else:
        raise NotImplementedError(type(model))

@torch.no_grad()
def run_awq(
    model, enc,
    w_bit, q_config,
    n_samples=512, seqlen=512,
    auto_scale=True, mse_range=True,
    # some configs for ablation study
    calib_data="pileval",
    dec = None
):
    from ..utils.calib_data import get_calib_dataset,get_vit_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name
    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module, is_enc_dec=False):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            # print(inp)
            # print(kwargs.items())
            raise ValueError  # early exit to break later inference
    
    inps = []
    layer_kwargs = {}
    
    awq_results = {
        "scale": [],
        "clip": [],
    }

    print(f"Model parameters have been saved to {file_path}")

    # 保存模型结构到日志文件
    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    # samples = torch.cat(samples, dim=0)


    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    
    # patch layer 0 to catch input and kwargs
    print(type(layers[0]))
    layers[0] = Catcher(layers[0])
    
    try:
        # layers[0](samples.to("cuda"))
        device = next(model.parameters()).device
        model(samples.to(device))
    except ValueError:  # work with early exit
       pass
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    del samples
    gc.collect()
    torch.cuda.empty_cache()
  

    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer, layer_kwargs,
                w_bit=w_bit, q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

        # Clear GPU memory
        torch.cuda.empty_cache()
        
        if mse_range:
            clip_list = auto_clip_block(layer,
                            w_bit=w_bit, q_config=q_config,
                            input_feat=input_feat,)
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
        
        # 2. 打开文件以写入模式
    file_path = "/zeng_chao/code/llm-awq/llizhi_awq.txt"
    with open(file_path, "w") as file:
        # 3. 遍历并写入参数
        for name, param in model.named_parameters():
            file.write(f"Parameter Name: {name}\n")
            file.write(f"Parameter Value:\n{param.data}\n")
            file.write("=" * 40 + "\n")

    print(f"Model parameters have been saved to {file_path}")
    return awq_results


def apply_awq(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])


@torch.no_grad()
def run_awq_T5(
    model, enc,
    w_bit, q_config,
    n_samples=512, seqlen=512,
    auto_scale=True, mse_range=True,
    # some configs for ablation study
    calib_data="pileval",
    dec = None
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name
    
    print(model)

    # 这些linear其实是一个个的encoder block / decoder block
    is_T5 = False
    if isinstance(model,T5ForConditionalGeneration):
        is_T5 = True
        labels = get_calib_dataset(
        data=calib_data, tokenizer=dec, n_samples=n_samples, block_size=seqlen)
        labels = torch.cat(labels, dim=0)
    layers = get_blocks(model)

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module, is_enc_dec=False):
            super().__init__()
            self.module = module
            self.is_enc_dec = is_enc_dec

        def forward(self, inp, **kwargs):
            if self.is_enc_dec:
                inps_dec.append(inp)
                layer_kwargs_dec.update(kwargs)
            else:
                inps.append(inp)
                layer_kwargs.update(kwargs)
            # print(inp)
            # print(kwargs.items())
            raise ValueError  # early exit to break later inference
    
    inps = []
    inps_dec = []
    layer_kwargs = {}
    layer_kwargs_dec = {}
    
    awq_results = {
        "scale": [],
        "clip": [],
    }
    # 保存模型结构到日志文件
    layers = get_blocks(model)
    layers2 = get_blocks2(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)


    layers[0] = layers[0].cuda()
    layers2[0] = layers2[0].cuda()
    move_embed(model, "cuda")
    
    # patch layer 0 to catch input and kwargs
    print(type(layers[0]))
    layers[0] = Catcher(layers[0])
    layers2[0] = Catcher(layers2[0])
    
    try:
        # layers[0](samples.to("cuda"))
        device = next(model.parameters()).device
        model(samples.to(device))
    except ValueError:  # work with early exit
       pass
    layers[0] = layers[0].module  # restore
    layers2[0] = layers2[0].module
    inps = inps[0]

    layers[0] = layers[0].cpu()
    layers2[0] = layers2[0].cpu()
    move_embed(model, "cpu")
    if is_T5:
        print('Encoder Decoder')
        # we actually dont need encoder args:
        model.encoder = model.encoder.cuda()
        move_embed(model, "cuda")
        # to get decoder args
        model.decoder.block[0].cuda()
        model.decoder.block[0] = Catcher(model.decoder.block[0], True)
        try:
            device = next(model.parameters()).device
            model(input_ids=samples.to(device),labels=labels.to(device))
        except ValueError:  # work with early exit
            pass
        model.decoder.block[0] = model.decoder.block[0].module
        inps_dec = inps_dec[0]
        # move embed back to cpu
        move_embed(model, "cpu")
        # move encoder back to cpu
        model.encoder = model.encoder.cpu()
        # move decoder.block[0] back to cpu
        model.decoder.block[0] = model.decoder.block[0].cpu()
    del samples
    gc.collect()
    torch.cuda.empty_cache()
    

    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ Encoder..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer, layer_kwargs,
                w_bit=w_bit, q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

        # Clear GPU memory
        torch.cuda.empty_cache()
        
        if mse_range:
            clip_list = auto_clip_block(layer,
                            w_bit=w_bit, q_config=q_config,
                            input_feat=input_feat,)
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
    
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers2)), desc="Running AWQ Decoder..."):
        layer = layers2[i]
        # print('#####################################')
        # print(layer)
        # print('#####################################')
        layer = layer.cuda()
        named_linears = get_named_linears(layer)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # for name in named_linears:
        #     print(name)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps_dec = inps_dec.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps_dec = layer(inps_dec, **layer_kwargs_dec)[0]
        
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        # print('=======================================')
        # for key, value in input_feat.items():
        #     print(key)
        # print('=======================================')

        # Clear GPU memory
        torch.cuda.empty_cache()

        if auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer, layer_kwargs_dec,
                w_bit=w_bit, q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers2[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

        # Clear GPU memory
        torch.cuda.empty_cache()
        
        if mse_range:
            clip_list = auto_clip_block(layer,
                            w_bit=w_bit, q_config=q_config,
                            input_feat=input_feat,)
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
        
    return awq_results

@torch.no_grad()
def run_awq_vit(
    model, modal_model, encoder,
    w_bit, q_config,
    n_samples=512, seqlen=512,
    auto_scale=True, mse_range=True,
    # some configs for ablation study
    calib_data="pileval",
):
    from ..utils.calib_data import get_calib_dataset,get_vit_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name
    print('------vit model------')
    print(model)
    print('-----vit model done-----')
    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module, is_enc_dec=False):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            print('-------------------------')
            print(inp)
            print(kwargs)
            print('-------------------------')
            inps.append(inp)
            layer_kwargs.update(kwargs)
            # print(inp)
            # print(kwargs.items())
            raise ValueError  # early exit to break later inference
    
    inps = []
    layer_kwargs = {}
    
    awq_results = {
        "scale": [],
        "clip": [],
    }

    # 保存模型结构到日志文件
    layers = get_blocks(model)

    samples = get_vit_calib_dataset(n_samples=n_samples, block_size=seqlen)
    # samples = torch.cat(samples, dim=0)
    print(samples.shape)

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    
    # patch layer 0 to catch input and kwargs
    print(type(layers[0]))
    layers[0] = Catcher(layers[0])
    
    try:
        # layers[0](samples.to("cuda"))
        device = next(model.parameters()).device
        with torch.cuda.amp.autocast(dtype=torch.float16):
            model(samples.to(device))
    except ValueError:  # work with early exit
       pass
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    del samples
    gc.collect()
    torch.cuda.empty_cache()
  
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
            layer = layers[i]
            layer = layer.cuda()
            named_linears = get_named_linears(layer)

            # firstly, get input features of all linear layers
            def cache_input_hook(m, x, y, name, feat_dict):
                x = x[0]
                x = x.detach().cpu()
                feat_dict[name].append(x)

            input_feat = defaultdict(list)
            handles = []
            for name in named_linears:
                handles.append(named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name,
                                    feat_dict=input_feat)))
            inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
            
            # import pdb
            # pdb.set_trace()
            # get output as next layer's input
            # print(inps.shape)
            inps = layer(inps, **layer_kwargs) #[0]
            for h in handles:
                h.remove()
            # now solve for scaling and clipping
            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

            # Clear GPU memory
            torch.cuda.empty_cache()

            if auto_scale:  # if it applies, we should also modify the input_feat with scales
                scales_list = auto_scale_block(
                    layer, layer_kwargs,
                    w_bit=w_bit, q_config=q_config,
                    input_feat=input_feat,
                )
                # apply_scale(layer, scales_list, input_feat_dict=input_feat)
                apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
                # append prefix to make names global
                awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

            # Clear GPU memory
            torch.cuda.empty_cache()
            
            if mse_range:
                clip_list = auto_clip_block(layer,
                                w_bit=w_bit, q_config=q_config,
                                input_feat=input_feat,)
                apply_clip(layer, clip_list)
                # append prefix to make names global
                awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

            layer = layer.cpu()
            # Haotian: check activation replacement
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()
            
            # 2. 打开文件以写入模式
        
    return awq_results
