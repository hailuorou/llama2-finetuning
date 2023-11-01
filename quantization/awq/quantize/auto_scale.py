import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from .modeling_chatglm import GLMBlock, RMSNorm

from .qmodule import ScaledActivation
from ..utils.module import get_op_by_name, get_op_name, set_op_by_name

import sys
sys.path.append('/zeng_chao/code/JiuTian/models')
from eva_vit_spatial import Block

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]
    
    scales = scales.to(ln.weight.device)

    # debugging start even scales = 1 does not work?
    """
    scales = scales * 0
    scales = scales + 1
    """
    # debugging end

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales, num_heads=None):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features
    
    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0
    
    # 支持荔枝
    # if fc1.out_features == fc2.in_features * 3:
    #     # fc1.weight.t_()
    #     # org_shape = fc1.weight.shape
    #     # fc1.weight.data = fc1.weight.data.reshape(org_shape[0] * num_heads, 3, -1)
    #     # value = fc1.weight.data[:, 2, :].reshape(org_shape[0], -1)
    #     # fc1.weight.data[:, 2, :] = value.div(scales.view(-1)).reshape(fc1.weight[:, 2, :].shape)
    #     # fc1.weight.data = fc1.weight.data.reshape(org_shape).t_()
        
    #     # if fc1.bias is not None:
    #     #     fc1.bias.data = fc1.bias.data.reshape(num_heads, 3, -1)
    #     #     value = fc1.bias.data[:, 2, :].reshape(-1)
    #     #     fc1.bias.data[:, 2, :] = value.div(scales.view(-1)).reshape(fc1.bias[:, 2, :].shape)
    #     #     fc1.bias.data = fc1.bias.data.reshape(-1)
    #     if fc1.out_features == 3*fc2.in_features:
    #             fc1.weight[-fc2.in_features:].div_(scales.view(-1, 1))
    #             if fc1.bias is not None:
    #                 fc1.bias[-fc2.in_features:].div_(scales.view(-1))

    # else:
    #     assert fc1.out_features == fc2.in_features
        
    #     fc1.weight.div_(scales.view(-1, 1))
    #     if fc1.bias is not None:
    #         fc1.bias.div_(scales.view(-1))

    # fc2.weight.mul_(scales.view(1, -1))

    # for p in fc1.parameters():
    #     assert torch.isnan(p).sum() == 0
    # for p in fc2.parameters():
    #     assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, nn.GELU) or isinstance(gelu, BloomGelu)
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0
    

@torch.no_grad()
def auto_scale_block(module, module_kwargs,
                     w_bit, q_config,
                     input_feat):
    from .quantizer import pseudo_quantize_tensor
    # firstly, get the weight quantize function
    if w_bit is not None:
        def w_quantize_func(p): return pseudo_quantize_tensor(
            p, n_bit=w_bit, **q_config,
        ).detach()
    else:
        def w_quantize_func(p): return p

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        # w: co, ci
        # x: n, ci
        weight = torch.cat([_m.weight for _m in linears2scale], dim=0)
        w_max = get_weight_scale(
            weight, q_group_size=q_config.get("q_group_size", -1))
        # Clear GPU memory
        del weight
        gc.collect()
        torch.cuda.empty_cache()

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)

        best_error = float('inf')
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = (x_max.pow(ratio) / w_max.pow(1-ratio)
                      ).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(
                    fc.weight.data) / (scales.view(1, -1))
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale(module2inspect, layers, inp, kwargs)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), scales)

    scales_list = []  # return the searched scales
    
    # import pdb
    # pdb.set_trace()

    if isinstance(module, OPTDecoderLayer):
        # attention input
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attn_layer_norm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))
        # attn out
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.out_proj],
            inp=input_feat['self_attn.out_proj'],
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.final_layer_norm,
            layers=[module.fc1],
            inp=input_feat['fc1'],
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.fc1,
            layers=[module.fc2],
            inp=input_feat['fc2'],
        ))
    elif isinstance(module, Block):
        # scales_list.append(_auto_get_scale(
        #     prev_op=module.norm1,
        #     layers=[module.attn.qkv],
        #     inp=input_feat['attn.qkv'],
        #     # module2inspect=module.self_attention, kwargs=module_kwargs,
        #     module2inspect=module, kwargs=module_kwargs,
        # ))
        # attn out
        scales_list.append(_auto_get_scale(
            prev_op=module.attn.qkv,
            layers=[module.attn.proj],
            inp=input_feat['attn.proj'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.norm2,
            layers=[module.mlp.fc1],
            inp=input_feat['mlp.fc1'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.fc1,
            layers=[module.mlp.fc2],
            inp=input_feat['mlp.fc2'],
        ))
    
    elif isinstance(module, GLMBlock):
        # attention input
        # print('---------------------------')
        # print('GLMBlock')
        # print('----------------------------')
        scales_list.append(_auto_get_scale(
            prev_op=module.input_layernorm,
            layers=[module.self_attention.query_key_value],
            inp=input_feat['self_attention.query_key_value'],
            # module2inspect=module.self_attention, kwargs=module_kwargs,
            module2inspect=module, kwargs=module_kwargs,
        ))
        # attn out
        # scales_list.append(_auto_get_scale(
        #     prev_op=module.self_attention.core_attention,
        #     layers=[module.self_attention.dense],
        #     inp=input_feat['self_attention.dense'],
        # ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.dense_h_to_4h,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))
    
    elif isinstance(module, LlamaDecoderLayer):
        # attention input
        scales_list.append(_auto_get_scale(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))
        # attn out
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.o_proj],
            inp=input_feat['self_attn.o_proj'],
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))
    elif isinstance(module, GPTNeoXLayer):
        # attention input
        # print('---------------------------')
        # print('GLMBlock')
        # print('----------------------------')
        scales_list.append(_auto_get_scale(
            prev_op=module.input_layernorm,
            layers=[module.attention.query_key_value],
            inp=input_feat['attention.query_key_value'],
            module2inspect=module.attention, kwargs=module_kwargs,
            # module2inspect=module, kwargs=module_kwargs,
        ))
    
        # # attn out
        scales_list.append(_auto_get_scale(
            prev_op=module.attention.query_key_value,
            layers=[module.attention.dense],
            inp=input_feat['attention.dense'],
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.dense_h_to_4h,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))
    
    elif isinstance(module, T5Block):
        # self_attention input
        self_kwargs = {}
        self_kwargs['position_bias'] = module_kwargs['position_bias']
        self_kwargs['layer_head_mask'] = module_kwargs['layer_head_mask']
        self_kwargs['mask'] = module_kwargs['attention_mask']
        self_kwargs['output_attentions'] = module_kwargs['output_attentions']
        scales_list.append(
                _auto_get_scale(prev_op=module.layer[0].layer_norm,
                layers=[module.layer[0].SelfAttention.q, 
                        module.layer[0].SelfAttention.k, 
                        module.layer[0].SelfAttention.v],
                inp=input_feat['layer.0.SelfAttention.q'],
                module2inspect=module.layer[0].SelfAttention, kwargs=self_kwargs
                )
        )
        # self_attention output
        if module.layer[0].SelfAttention.v.weight.shape == module.layer[0].SelfAttention.o.weight.shape:
            scales_list.append(
                    _auto_get_scale(prev_op=module.layer[0].SelfAttention.v,
                    layers=[module.layer[0].SelfAttention.o],
                    inp=input_feat['layer.0.SelfAttention.o'],
                    )
            )
        if module.is_decoder:
            # print('@@@@@@@@@@@@@@@@@@进入is_decoder')
            # cross_attention input
            index1 = 'layer.2.DenseReluDense.wi_0'
            index2 = 'layer.2.DenseReluDense.wo'
            cross_kwargs = {}
            cross_kwargs['mask'] = module_kwargs['encoder_attention_mask']
            cross_kwargs['key_value_states'] = module_kwargs['encoder_hidden_states']
            cross_kwargs['output_attentions'] = module_kwargs['output_attentions']
            cross_kwargs['position_bias'] = module_kwargs['encoder_decoder_position_bias']
            scales_list.append(
                    _auto_get_scale(prev_op=module.layer[1].layer_norm,
                    layers=[module.layer[1].EncDecAttention.q, 
                            module.layer[1].EncDecAttention.k, 
                            module.layer[1].EncDecAttention.v],
                    inp=input_feat['layer.1.EncDecAttention.q'],
                    module2inspect=module.layer[1].EncDecAttention, kwargs=cross_kwargs
                    )
            )
            # cross_attention output
            if module.layer[1].EncDecAttention.v.weight.shape == module.layer[1].EncDecAttention.o.weight.shape:
                scales_list.append(
                        _auto_get_scale(prev_op=module.layer[1].EncDecAttention.v,
                        layers=[module.layer[1].EncDecAttention.o],
                        inp=input_feat['layer.1.EncDecAttention.o'],
                        )
                )
        
        else:
            index1 = 'layer.1.DenseReluDense.wi_0'
            index2 = 'layer.1.DenseReluDense.wo'
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.layer[-1].layer_norm,
            layers=[module.layer[-1].DenseReluDense.wi_0,
                    module.layer[-1].DenseReluDense.wi_1],
            inp=input_feat[index1],
            module2inspect=module.layer[-1].DenseReluDense
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.layer[-1].DenseReluDense.wi_1,
            layers=[module.layer[-1].DenseReluDense.wo],
            inp=input_feat[index2],
        ))
        
    elif isinstance(module, BloomBlock):
        # attention input
        scales_list.append(_auto_get_scale(
            prev_op=module.input_layernorm,
            layers=[module.self_attention.query_key_value],
            inp=input_feat['self_attention.query_key_value'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.gelu_impl,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))
    elif "mpt" in str(module.__class__).lower():
        # attention input
        scales_list.append(_auto_get_scale(
            prev_op=module.norm_1,
            layers=[module.attn.Wqkv],
            inp=input_feat['attn.Wqkv'],
            module2inspect=module.attn, 
            kwargs=module_kwargs,
        ))
        
        # attn out
        scales_list.append(_auto_get_scale(
            prev_op=module.attn.Wqkv,
            layers=[module.attn.out_proj],
            inp=input_feat['attn.out_proj'],
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.norm_2,
            layers=[module.ffn.up_proj],
            inp=input_feat['ffn.up_proj'],
            module2inspect=module.ffn,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.ffn.act,
            layers=[module.ffn.down_proj],
            inp=input_feat['ffn.down_proj'],
        ))

    elif "falcon" in str(module.__class__).lower():         
        # attn out
        # Haotian: TBD: need to handle repeated scales for MQ
        """ 
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1, as long as it is scaled, everything is screwed up
        if "falcon-7b" in str(module.__class__).lower():
            scales_list.append(_auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.mlp.dense_h_to_4h, module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))
        elif "falcon-40b" in str(module.__class__).lower():
            scales_list.append(_auto_get_scale(
                prev_op=module.ln_attn,
                layers=[module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))
            scales_list.append(_auto_get_scale(
                prev_op=module.ln_mlp,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat['mlp.dense_h_to_4h'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))
        else:
            raise NotImplementedError("Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported")
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.act,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))
    
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list

def apply_scale(module, scales_list, input_feat_dict=None):
    # print('++++++++++++++++++++++++++')
    # for prev_op_name, layer_names, scales in scales_list:
    #     print(prev_op_name)
    #     print(layer_names)
    #     print(scales)
    #     print('----------------------')
    # print(module.num_attention_heads if hasattr(module, 'num_attention_heads') else None)
    # print('+++++++++++++++++++++++++++')
    # exit()
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()
        
        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
            # 支持荔枝
            # scale_fc_fc(prev_op, layers[0], scales, module.num_attention_heads if hasattr(module, 'num_attention_heads') else 64)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm, RMSNorm, T5LayerNorm)):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, nn.GELU) or isinstance(prev_op, BloomGelu):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(
                f"prev_op {type(prev_op)} not supported yet!")
            
        # scale_ln_fcs(prev_op, layers, scales)
            
        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:  
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
