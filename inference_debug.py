import os
import gc
import json
import argparse
import torch
import warnings
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete

def main(args):
    # Add debug flag
    debug = getattr(args, 'debug', False)
    
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    cot_name = "cot" if args.with_cot else "direct"
    load_adapter_path = os.path.join(
        ROOT_DIR, 
        "offline", 
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
    
    # DEBUG: Print base adapter path
    if debug:
        print("\n" + "="*80)
        print("DEBUG: LoRA适配器加载路径信息")
        print("="*80)
        print(f"基础路径: {load_adapter_path}")
        print(f"路径存在: {os.path.exists(load_adapter_path)}")
        if os.path.exists(load_adapter_path):
            print(f"路径内容: {os.listdir(load_adapter_path)}")
        print("="*80 + "\n")
    
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output",
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method, 
    )
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        
        # Only debug first few samples to avoid too much output
        debug_samples = 3 if debug else 0
        
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs):
                text = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, 
                                        passages=psgs)
                pred = {
                    "test_id": test_id, 
                    "question": question, 
                    "answer": answer, 
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            else:
                # DEBUG: Print detailed information for first few samples
                if debug and test_id < debug_samples:
                    print(f"\n{'='*80}")
                    print(f"DEBUG: 样本 {test_id} 的适配器加载详情")
                    print(f"{'='*80}")
                    print(f"问题: {question[:80]}...")
                    print(f"Passages数量: {len(passages)}")
                
                # Load adapters for each passage
                # IMPORTANT: Don't specify adapter_name to match how they were saved during encoding
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
                    
                    # DEBUG: Check adapter path
                    if debug and test_id < debug_samples:
                        print(f"\n--- Passage {pid} ---")
                        print(f"适配器路径: {adapter_path}")
                        print(f"路径存在: {os.path.exists(adapter_path)}")
                        
                        if os.path.exists(adapter_path):
                            files = os.listdir(adapter_path)
                            print(f"路径内文件: {files}")
                            
                            # Check adapter file size
                            adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
                            if os.path.exists(adapter_file):
                                size = os.path.getsize(adapter_file)
                                print(f"adapter_model.safetensors 大小: {size:,} bytes ({size/1024:.2f} KB)")
                            else:
                                print("❌ adapter_model.safetensors 不存在！")
                        else:
                            print("❌ 适配器路径不存在！")
                    
                    if pid == 0:
                        # Load first adapter without specifying adapter_name
                        try:
                            model = PeftModel.from_pretrained(
                                model, 
                                adapter_path,
                                is_trainable = False
                            )
                            # Get the default adapter name that was assigned
                            first_adapter_name = model.active_adapter if hasattr(model, 'active_adapter') else "default"
                            
                            if debug and test_id < debug_samples:
                                print(f"✓ 成功加载第一个适配器")
                                print(f"  分配的适配器名称: {first_adapter_name}")
                                
                        except Exception as e:
                            if debug and test_id < debug_samples:
                                print(f"❌ 加载失败: {e}")
                            raise
                    else:
                        # Load additional adapters with unique names
                        try:
                            model.load_adapter(adapter_path, adapter_name = f"adapter_{pid}")
                            if debug and test_id < debug_samples:
                                print(f"✓ 成功加载适配器 'adapter_{pid}'")
                        except Exception as e:
                            if debug and test_id < debug_samples:
                                print(f"❌ 加载失败: {e}")
                            raise
                
                # Get list of all loaded adapter names
                if hasattr(model, 'peft_config'):
                    adapter_names = list(model.peft_config.keys())
                else:
                    # Fallback: assume default naming
                    adapter_names = ["default"] + [f"adapter_{i}" for i in range(1, len(passages))]
                
                if debug and test_id < debug_samples:
                    print(f"\n已加载的所有适配器: {adapter_names}")
                    print(f"将要合并的适配器: {adapter_names[:len(passages)]}")
                
                # merge
                # Use linear combination instead of cat for better compatibility
                # Normalize weights so each adapter contributes equally
                num_adapters = len(passages)
                try:
                    model.add_weighted_adapter(
                        adapters = adapter_names[:num_adapters], 
                        weights = [1.0 / num_adapters] * num_adapters,
                        adapter_name = "merge", 
                        combination_type = "linear",
                    )
                    if debug and test_id < debug_samples:
                        print(f"✓ 成功合并适配器为 'merge'")
                        print(f"  权重: {[1.0 / num_adapters] * num_adapters}")
                        print(f"  合并方式: linear")
                except Exception as e:
                    if debug and test_id < debug_samples:
                        print(f"❌ 合并失败: {e}")
                    raise
                
                model.set_adapter("merge")
                
                # Verify adapter is active (helps with debugging)
                if hasattr(model, 'active_adapter'):
                    current_adapter = model.active_adapter
                    if debug and test_id < debug_samples:
                        print(f"当前激活的适配器: {current_adapter}")
                    if current_adapter != "merge":
                        warnings.warn(f"Expected adapter 'merge' but got '{current_adapter}'")
                
                if debug and test_id < debug_samples:
                    # Check if model has non-zero adapter parameters
                    total_params = 0
                    adapter_params = 0
                    for name, param in model.named_parameters():
                        total_params += 1
                        if 'lora' in name.lower():
                            adapter_params += 1
                    print(f"\n模型参数统计:")
                    print(f"  总参数数量: {total_params}")
                    print(f"  LoRA参数数量: {adapter_params}")
                    
                    # Check if adapters have non-zero values
                    has_nonzero = False
                    for name, param in model.named_parameters():
                        if 'lora' in name.lower() and param.abs().sum().item() > 0:
                            has_nonzero = True
                            break
                    print(f"  LoRA参数非零: {has_nonzero}")
                    print(f"{'='*80}\n")
                
                ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)  
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True, choices=["icl", "prag", "combine"])
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    # Debug flag
    parser.add_argument("--debug", action="store_true", help="Print detailed debugging information")
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
