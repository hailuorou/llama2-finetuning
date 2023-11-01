from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model, load_checkpoint_in_model
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq,run_awq_vit
from awq.quantize.quantizer import pseudo_quantize_model_weight, real_quantize_model_weight
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model

from awq.quantize.modeling_chatglm import ChatGLMForConditionalGeneration,ChatGLMModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from awq.CustomTokenizer import TiktokenTokenizer

## from vllm.model_executor.models.chatglm import ChatGLMModel
import sys
sys.path.append('/zeng_chao/code/JiuTian/models')
from JiuTian import JiuTian


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path of the hf model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
# model config
parser.add_argument('--parallel', action='store_true',
                    help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument('--max_memory', type=str, nargs='*',
                    help="List of device_id:max_memory pairs to be parsed into a dictionary; " \
                        + "Example: 0:10GiB 1:10GiB cpu:30GiB; " \
                        + "mode details here: " \
                        + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling")
parser.add_argument('--auto_parallel', action='store_true',
                    help="automatically set parallel and batch_size")
# quantization config
parser.add_argument('--w_bit', type=int, default=None)
parser.add_argument('--q_group_size', type=int, default=-1)
parser.add_argument('--no_zero_point', action='store_true',
                    help="disable zero_point")
parser.add_argument('--q_backend', type=str,
                    default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument('--dump_quant', type=str, default=None,
                    help='save quantized model')
parser.add_argument('--load_quant', type=str, default=None,
                    help='load quantized model')
# apply/save/load awq
parser.add_argument('--run_awq', action='store_true',
                    help="perform awq search process")
parser.add_argument('--dump_awq', type=str, default=None,
                    help="save the awq search results")
parser.add_argument('--load_awq', type=str, default=None,
                    help="load the awq search results")
args = parser.parse_args()

max_memory = [v.split(':') for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k):v for k,v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization

}
print("Quantization config:", q_config)

# build model and tokenizer

def build_model_and_enc(model_path):
    # if not os.path.exists(model_path):  # look into ssd
    #     raise FileNotFoundError(f"{model_path} not found!")
    # print(f"* Building model {model_path}")

    # # all hf model
    # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # if "mpt" in config.__class__.__name__.lower():
    #     enc = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    # else:
    #     enc = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        # enc = TiktokenTokenizer(model_max_length=200, padding_side='left') # 加载lizhi模型得tokenizer
        # enc = T5Tokenizer.from_pretrained("data/llm/flan-t5-xl", use_fast=False, trust_remote_code=True)
        # dec = T5Tokenizer.from_pretrained("data/llm/flan-t5-xl", truncation_side='right')

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            # model = AutoModelForCausalLM.from_config(config=config,
            #                                          torch_dtype=torch.float16, trust_remote_code=True)
            model = ChatGLMModel.from_pretrained("/zhang_miao/zeng_chao/codes/llm-awq/quant_cache/chatglm2-6b-w4-g128-awq.pt", config=config, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True)
        
        model.tie_weights()
        
        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer", "ModuleList"],
            **kwargs
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        # model = ChatGLMModel.from_pretrained(
        #    model_path, config=config, trust_remote_code=True, **kwargs)
        # model = ChatGLMModel.load_weights(
        #      model_path)
        # config = AutoConfig.from_pretrained(model_path=model_path, trust_remote_code=True)
        # engine_args = EngineArgs(
        #     model=model_path,
        #     tokenizer=enc,
        #     tokenizer_mode="auto",
        #     trust_remote_code=True,
        #     tensor_parallel_size=1,
        #     dtype="auto",
        #     seed=0,
        # )
        # torch.distributed.init_process_group(
        #     backend="nccl",
        #     world_size=1,
        #     rank=0,
        #     init_method="tcp://localhost:50007",
        # )
        # torch.distributed.all_reduce(torch.zeros(1).cuda())
        # initialize_model_parallel(1,1)
        # model = ChatGLMModel.from_pretrained(pretrained_model_name_or_path=model_path)
        # model = ChatGLMModel(config=config)
        # model.load_weights(model_path, None, False)
        # ChatGLMModel._load_from_state_dict()
        
        # model = T5ForConditionalGeneration.from_pretrained(
        #     model_path, config=config, trust_remote_code=True, **kwargs)
        print('jiutian loading.....')
        modal_model = JiuTian(vit_model="/zeng_chao/code/JiuTian/eva_vit_g.pth",
                    q_former_model="/zeng_chao/code/JiuTian/instruct_blip_vicuna13b_trimmed.pth",
                    llm_model="/zeng_chao/dataset/llmWeight/chatglm2-6b", # "/zhang_miao_user1/JiuTian/chatglm2-6b",
                    ckpt="/zeng_chao/code/JiuTian/JiuTian_Spatial0821.pth").to('cuda')
        print('jiutian load done')
        modal_model.eval()
        model=modal_model.visual_encoder
        
        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"
                        
            awq_results = run_awq_vit(
                model, modal_model, modal_model.visual_encoder,
                w_bit=args.w_bit, q_config=q_config,
                n_samples=128, seqlen=512
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)
                
                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)
                
            exit(0)
                
        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)
        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert args.dump_quant is None, \
                    "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )
                # print("*************************************")
                # print(model)
                # print('*************************************')
                # model()
                
                if args.dump_quant:
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)
                    print(
                        f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError
            
        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"],
            **kwargs
        )
        model = dispatch_model(model, device_map=device_map)

    return model


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    # a hack here to auto set model group
    model= build_model_and_enc(args.model_path)
    

    if args.tasks is not None:
        task_names = args.tasks.split(",")

        lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=args.batch_size,
            no_cache=True,
            num_fewshot=args.num_fewshot,
        )

        print(evaluator.make_table(results))

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
