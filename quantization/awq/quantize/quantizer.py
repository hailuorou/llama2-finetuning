import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name,get_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers import T5Tokenizer, T5ForConditionalGeneration

from .modeling_chatglm import GLMBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif 'mptblock' in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif 'falcon' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=8,
                           zero_point=True, q_group_size=-1,
                           inplace=False,
                           get_scale_zp=False
                           ):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

@torch.no_grad()
def pseudo_quantize_model_weight(
    model, w_bit, q_config,
):    
    from .pre_quant import get_blocks, get_named_linears, get_blocks2
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, **q_config)
            m.cpu()
    # if isinstance(model, T5ForConditionalGeneration):
    #     layers = get_blocks2(model)
    #     for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
    #         named_linears = get_named_linears(layers[i])
    #         for n, m in named_linears.items():
    #             m.cuda()
    #             m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, **q_config)
    #             m.cpu()


@torch.no_grad()
def real_quantize_model_weight(
    model, w_bit, q_config,
    init_only=False
):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears,get_blocks2
    assert q_config["zero_point"], "We only support zero_point quantization now."
    
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="real weight quantization..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        # print('++++++++++++++++++++++++++++++')
        # for name, module in named_linears.items():
        #     print(name)
        #     print(module)
        # print('+++++++++++++++++++++++++++++++')
        scale_activations(layer)
        # 2. 打开文件以写入模式
        # file_path = "/zeng_chao/code/llm-awq/llizhi_layer.txt"
        # with open(file_path, "a") as file:
        #     # 3. 遍历并写入参数
        #     for name, param in layer.named_parameters():
        #         file.write(f"Parameter Name: {name}\n")
        #         file.write(f"Parameter Value:\n{param.data}\n")
        #         file.write("=" * 40 + "\n")

        # print(f"Model parameters have been saved to {file_path}")
        
        for name, module in named_linears.items():
            # print('*******************************')
            # print(init_only)
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config['q_group_size'], True)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
        # 2. 打开文件以写入模式
                # file_path = "/zeng_chao/code/llm-awq/llizhi_q_linear.txt"
                # with open(file_path, "a") as file:
                #     # 3. 遍历并写入参数
                #     for name, param in q_linear.named_parameters():
                #         file.write(f"Parameter Name: {name}\n")
                #         file.write(f"Parameter Value:\n{param.data}\n")
                #         file.write("=" * 40 + "\n")

                # print(f"Model parameters have been saved to {file_path}")
            else:
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config)
                # print('----------------------------')
                # print(module.weight.data)
                # print('----------------------------')
                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config['q_group_size'], False, scales, zeros)
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                #2. 打开文件以写入模式
                # file_path = "/zeng_chao/code/llm-awq/llizhi_q_linear.txt"
                # with open(file_path, "a") as file:
                #     # 3. 遍历并写入参数
                #     for name, param in q_linear.named_parameters():
                #         file.write(f"Parameter Name: {name}\n")
                #         file.write(f"Parameter Value:\n{param.data}\n")
                #         file.write("=" * 40 + "\n")
                # print('---------------')
                # print(q_linear)
                # print('---------------')

                #print(f"Model parameters have been saved to {file_path}")
                torch.cuda.empty_cache()
                gc.collect()
    if isinstance(model, T5ForConditionalGeneration):
        layers = get_blocks2(model)
        for i in tqdm(range(len(layers)), desc="real decoder weight quantization..." + ("(init only)" if init_only else "")):
            layer = layers[i]
            named_linears = get_named_linears(layer)
            scale_activations(layer)

            for name, module in named_linears.items():
                if init_only:
                    q_linear = WQLinear.from_linear(
                        module, w_bit, q_config['q_group_size'], True)
                    q_linear.to(next(layer.parameters()).device)
                    set_op_by_name(layer, name, q_linear)
                else:
                    module.cuda()
                    module.weight.data, scales, zeros = pseudo_quantize_tensor(module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config)
                    scales = scales.t().contiguous()
                    zeros = zeros.t().contiguous()
                    q_linear = WQLinear.from_linear(
                        module, w_bit, q_config['q_group_size'], False, scales, zeros)
                    module.cpu()
                    q_linear.to(next(layer.parameters()).device)
                    set_op_by_name(layer, name, q_linear)
                    torch.cuda.empty_cache()
                    gc.collect()

    print(model)
    # 遍历模型的每一层并输出参数
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Size: {param.size()}, Type: {param.dtype}")
    # 指定要保存的文件名
    file_path = "model_parameters.pt"

    # 保存模型的参数
    torch.save(model.state_dict(), file_path)     
                    
    torch.cuda.empty_cache()
    gc.collect()

        # 2. 打开文件以写入模式
    # file_path = "/zeng_chao/code/llm-awq/llizhi_quantizer.txt"
    # with open(file_path, "w") as file:
    #     # 3. 遍历并写入参数
    #     for name, param in model.named_parameters():
    #         file.write(f"Parameter Name: {name}\n")
    #         file.write(f"Parameter Value:\n{param.data}\n")
    #         file.write("=" * 40 + "\n")

    # print(f"Model parameters have been saved to {file_path}"