import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete


def merge_adapters_into_base(peft_model, adapter_names, weights):
    """
    Merge multiple LoRA adapters into the base model weights.
    
    This function directly merges the weighted sum of all adapter contributions
    into the base model weights, avoiding bugs in PEFT's add_weighted_adapter.
    
    Args:
        peft_model: PeftModel with loaded adapters
        adapter_names: list of adapter names to merge  
        weights: list of weights for each adapter (should sum to desired total contribution)
    """
    # Iterate over all modules and merge LoRA layers
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            # Accumulate weighted delta W = sum(weight_i * scaling_i * B_i @ A_i)
            delta_weight = None
            
            for adapter_name, weight in zip(adapter_names, weights):
                if adapter_name not in module.lora_A:
                    continue
                    
                lora_A = module.lora_A[adapter_name].weight
                lora_B = module.lora_B[adapter_name].weight
                scaling = module.scaling[adapter_name]
                
                # Compute this adapter's contribution: weight * scaling * (B @ A)
                adapter_delta = weight * scaling * (lora_B @ lora_A)
                
                if delta_weight is None:
                    delta_weight = adapter_delta
                else:
                    delta_weight = delta_weight + adapter_delta
            
            # Add the merged delta to the base weight
            if delta_weight is not None and hasattr(module, 'base_layer'):
                # Get the base layer (could be Linear, Embedding, etc.)
                base_layer = module.base_layer
                if hasattr(base_layer, 'weight'):
                    # Only convert dtype if needed
                    if delta_weight.dtype != base_layer.weight.dtype:
                        delta_weight = delta_weight.to(base_layer.weight.dtype)
                    base_layer.weight.data += delta_weight


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
    
    # For prag and combine modes, we need to restore the model after each sample
    # Store the original state dict for restoration (deep copy to avoid reference issues)
    if args.inference_method != "icl":
        original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
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

            def get_pred(mdl, psgs):
                text = predict(mdl, tokenizer, generation_config, 
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
                # Load all passage-specific adapters
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
                    if pid == 0:
                        peft_model = PeftModel.from_pretrained(
                            model, 
                            adapter_path,
                            adapter_name = "0", 
                            is_trainable = False
                        )
                    else:
                        peft_model.load_adapter(adapter_path, adapter_name = str(pid)) 
                
                # Merge adapters into base model weights
                adapter_names = [str(i) for i in range(len(passages))]
                weights = [1.0] * len(passages)
                
                # Merge all adapters into base model weights (modifies in-place)
                merge_adapters_into_base(peft_model, adapter_names, weights)
                
                # Get the base model with merged weights for inference
                merged_model = peft_model.get_base_model()
                
                # Run inference with merged model
                ret.append(get_pred(merged_model, psgs=None if args.inference_method == "prag" else passages))
                
                # Unload adapters and restore original model weights
                peft_model.unload()
                model.load_state_dict(original_state_dict)
                
                # Clean up memory
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
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)