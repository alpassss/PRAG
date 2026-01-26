import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete

# Import debug utilities
from lora_debug import (
    debug_inference_load_adapter,
    debug_inference_after_merge,
    capture_base_model_weights,
    get_lora_weights,
    print_lora_weight_summary,
    print_adapter_info,
    print_lora_storage_info,
    compare_model_weights_before_after_lora,
    read_safetensors_file,
)

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
    
    # DEBUG: Show load adapter path
    if args.debug:
        print("\n" + "#" * 80)
        print(" DEBUG: INFERENCE - INITIALIZATION")
        print("#" * 80)
        print(f"  Inference method: {args.inference_method}")
        print_lora_storage_info(load_adapter_path, "LoRA Adapter Root Path")
    
    # Capture base model weights for comparison (only needed for prag/combine modes)
    base_model_weights = None
    if args.debug and args.inference_method != "icl":
        base_model_weights = capture_base_model_weights(model)
        print(f"  Captured {len(base_model_weights)} base model weight tensors for comparison")
    
    debug_count = 0  # Only debug first few iterations to avoid too much output
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
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]
            
            # Only debug first 2 iterations
            should_debug = args.debug and debug_count < 2

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
                if should_debug:
                    print("\n" + "#" * 80)
                    print(f" DEBUG: INFERENCE - LOADING ADAPTERS (test_id={test_id})")
                    print("#" * 80)
                
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
                    if pid == 0:
                        model = PeftModel.from_pretrained(
                            model, 
                            adapter_path,
                            adapter_name="0", 
                            is_trainable=False
                        )
                    else:
                        model.load_adapter(adapter_path, adapter_name=str(pid))
                    
                    # DEBUG: Show loaded adapter info
                    if should_debug:
                        debug_inference_load_adapter(model, adapter_path, str(pid), pid)
                
                # merge
                if should_debug:
                    print("\n" + "#" * 80)
                    print(f" DEBUG: INFERENCE - MERGING ADAPTERS (test_id={test_id})")
                    print("#" * 80)
                    print(f"  Adapters to merge: {[str(i) for i in range(len(passages))]}")
                    print(f"  Weights: {[1] * len(passages)}")
                    print(f"  Combination type: cat")
                
                model.add_weighted_adapter(
                    adapters = [str(i) for i in range(len(passages))], 
                    weights = [1] * len(passages),
                    adapter_name = "merge", 
                    combination_type = "cat",
                )
                model.set_adapter("merge")
                
                # DEBUG: Show merged adapter info and compare with base model
                if should_debug:
                    debug_inference_after_merge(
                        model, 
                        [str(i) for i in range(len(passages))],
                        base_model_weights
                    )
                    debug_count += 1
                
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
    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable LoRA weight debugging output")
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)