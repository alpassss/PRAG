"""
Parametric RAG Inference Module

This module implements three inference modes following the PRAG paper:
- ICL: Traditional In-Context Learning RAG (passages in prompt, no adapter)
- PRAG: Parametric RAG (adapters merged into model, no passages in prompt)  
- Combine: Both adapters and passages used together

The key is loading all adapters and then merging them properly.
"""

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


def load_and_merge_all_adapters(base_model, adapter_paths):
    """
    Load multiple LoRA adapters and merge them all into the base model.
    
    This loads all adapters first (giving each a unique name), then sets them
    all as active, and finally merges them into the base weights.
    
    Args:
        base_model: The base model
        adapter_paths: List of paths to adapter directories
        
    Returns:
        A new model with all adapter weights merged into base weights
    """
    if not adapter_paths:
        return base_model
    
    # Load the first adapter
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[0],
        adapter_name="adapter_0",
        is_trainable=False
    )
    
    # Load remaining adapters
    for i, adapter_path in enumerate(adapter_paths[1:], start=1):
        peft_model.load_adapter(adapter_path, adapter_name=f"adapter_{i}")
    
    # Set all adapters as active
    adapter_names = [f"adapter_{i}" for i in range(len(adapter_paths))]
    peft_model.set_adapter(adapter_names)
    
    # Merge all active adapters into base model and unload
    merged_model = peft_model.merge_and_unload()
    
    # Clean up
    del peft_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return merged_model


def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    
    # Load the base model
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
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
    # Store the original state dict for restoration
    if args.inference_method != "icl":
        print("Saving original model state for restoration...")
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

            if args.inference_method == "icl":
                # ICL mode: Use base model with passages in prompt
                text = predict(model, tokenizer, generation_config, 
                              question, with_cot=args.with_cot, 
                              passages=passages)
            else:
                # PRAG or Combine mode: Load and merge adapters
                adapter_paths = []
                for pid in range(len(passages)):
                    adapter_path = os.path.join(
                        load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}"
                    )
                    adapter_paths.append(adapter_path)
                
                # Load and merge all adapters
                merged_model = load_and_merge_all_adapters(model, adapter_paths)
                
                # Generate prediction
                if args.inference_method == "prag":
                    # PRAG mode: No passages in prompt (knowledge is in parameters)
                    text = predict(merged_model, tokenizer, generation_config,
                                  question, with_cot=args.with_cot,
                                  passages=None)
                else:  # combine mode
                    # Combine mode: Use both adapters AND passages
                    text = predict(merged_model, tokenizer, generation_config,
                                  question, with_cot=args.with_cot,
                                  passages=passages)
                
                # Restore original model weights for next sample
                model.load_state_dict(original_state_dict)
                
                # Clean up
                del merged_model
                torch.cuda.empty_cache()
                gc.collect()
            
            # Build prediction record
            pred = {
                "test_id": test_id, 
                "question": question, 
                "answer": answer, 
                "text": text,
            }
            pred.update(evaluate(text, answer, args.with_cot))
            ret.append(pred)

        # Save predictions
        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

        # Calculate and save metrics
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)
        
        print(f"Results for {filename}:")
        print(ret_str.split("\n\n")[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametric RAG Inference")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name: llama3-8b-instruct, qwen2.5-1.5b-instruct, llama3.2-1b-instruct")
    parser.add_argument("--max_new_tokens", type=int, required=True,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name: hotpotqa, 2wikimultihopqa, popqa, complexwebquestions")
    parser.add_argument("--data_type", type=str, default=None,
                       help="Data type within dataset (optional)")
    parser.add_argument("--with_cot", action="store_true",
                       help="Use Chain-of-Thought prompting")
    parser.add_argument("--sample", type=int, default=-1,
                       help="Number of samples to process (-1 for all)")
    parser.add_argument("--augment_model", type=str, default=None,
                       help="Model used for data augmentation")
    parser.add_argument("--num_train_epochs", type=int, required=True,
                       help="Number of training epochs used in encode step")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate used in encode step")
    parser.add_argument("--inference_method", type=str, required=True, 
                       choices=["icl", "prag", "combine"],
                       help="Inference method: icl (RAG), prag (Parametric RAG), combine (both)")
    parser.add_argument("--lora_rank", type=int, required=True,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, required=True,
                       help="LoRA alpha")
    
    args = parser.parse_args()
    
    if args.augment_model is None:
        args.augment_model = args.model_name
    
    print("=" * 60)
    print("Parametric RAG Inference")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Inference Method: {args.inference_method}")
    print(f"LoRA Config: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print("=" * 60)
    
    main(args)