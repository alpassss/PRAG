import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete, print_lora_weights, print_adapter_weights_from_path, compare_lora_weights, DEBUG_MAX_ITEMS, print_lora_sample_values, print_merged_lora_info, compare_base_model_with_lora

def main(args):
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
    
    # Debug: Print adapter path info
    if args.debug_lora:
        print(f"\n[DEBUG] Load adapter path: {load_adapter_path}")
        print(f"[DEBUG] Path exists: {os.path.exists(load_adapter_path)}")
        
        # Also check base weight path for comparison
        base_weight_path = os.path.join(
            ROOT_DIR, 
            "offline", 
            args.model_name, 
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            "base_weight",
        )
        print(f"[DEBUG] Base weight path: {base_weight_path}")
        if os.path.exists(base_weight_path):
            print_adapter_weights_from_path(base_weight_path, tag="BASE weights for reference")
    
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
        
        # Debug counter to limit verbose output
        debug_count = 0
        
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]
            
            # Enable debug logging for first few items
            enable_debug = args.debug_lora and debug_count < DEBUG_MAX_ITEMS

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
                if enable_debug:
                    print(f"\n{'='*60}")
                    print(f"[DEBUG] INFERENCE - Processing test_id={test_id}")
                    print(f"{'='*60}")
                    print(f"[DEBUG] Question: {question[:80]}...")
                    print(f"[DEBUG] Number of passages/adapters to load: {len(passages)}")
                
                # Store reference to base model for comparison later
                base_model_ref = model
                
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
                    
                    if enable_debug:
                        print(f"\n[DEBUG] --- Loading Adapter {pid} ---")
                        print(f"[DEBUG] Loading from path: {adapter_path}")
                        if os.path.exists(adapter_path):
                            print_lora_sample_values(adapter_path, tag=f"Adapter {pid} (from file)", layer_idx=0)
                        else:
                            print(f"WARNING: Adapter path does not exist: {adapter_path}")
                    
                    if pid == 0:
                        model = PeftModel.from_pretrained(
                            model, 
                            adapter_path,
                            adapter_name = "0", 
                            is_trainable = False
                        )
                        
                        if enable_debug:
                            print(f"[DEBUG] Loaded first adapter (adapter_name='0')")
                            print_lora_sample_values(model, tag=f"Adapter 0 (in model)", layer_idx=0)
                    else:
                        model.load_adapter(adapter_path, adapter_name = str(pid)) 
                        
                        if enable_debug:
                            print(f"[DEBUG] Loaded additional adapter (adapter_name='{pid}')")
                
                # Debug: Print all adapter info before merge
                if enable_debug:
                    print(f"\n[DEBUG] --- All Adapters Loaded ---")
                    print(f"[DEBUG] Available adapters: {list(model.peft_config.keys())}")
                
                # merge
                adapter_names = [str(i) for i in range(len(passages))]
                if enable_debug:
                    print(f"\n[DEBUG] --- Merging Adapters ---")
                    print(f"[DEBUG] Merging adapters: {adapter_names}")
                    print(f"[DEBUG] Merge weights: {[1] * len(passages)}")
                    print(f"[DEBUG] Combination type: 'cat'")
                
                model.add_weighted_adapter(
                    adapters = adapter_names, 
                    weights = [1] * len(passages),
                    adapter_name = "merge", 
                    combination_type = "cat",
                )
                model.set_adapter("merge")
                
                if enable_debug:
                    print(f"\n[DEBUG] --- After Merge ---")
                    print(f"[DEBUG] Active adapter: {model.active_adapter}")
                    print_merged_lora_info(model, adapter_names, tag="MERGED adapters")
                    print_lora_sample_values(model, tag="MERGED adapter (sample values)", layer_idx=0)
                    
                    # Compare base model with LoRA-applied model
                    print(f"\n[DEBUG] --- Base Model vs LoRA Model Comparison ---")
                    compare_base_model_with_lora(base_model_ref, model, tag="After merging adapters")
                
                # Debug: Show what input will be used
                if enable_debug:
                    if args.inference_method == "prag":
                        print(f"\n[DEBUG] PRAG mode: Using LoRA weights WITHOUT context in prompt")
                    else:
                        print(f"\n[DEBUG] COMBINE mode: Using LoRA weights WITH context in prompt")
                
                ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
                
                debug_count += 1

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
    # Debug
    parser.add_argument("--debug_lora", action="store_true",
                        help="Enable LoRA weight debugging: print A/B matrix values during loading and merging")
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)