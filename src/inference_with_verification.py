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


def compare_model_parameters(base_model, modified_model, sample_layers=3):
    """Compare parameters between base and modified model to verify LoRA is active
    
    Note: LoRA adds new parameters (lora_A, lora_B) to the model. We need to compare
    the base model parameters to see if they've been modified by LoRA's influence.
    """
    base_params = dict(base_model.named_parameters())
    modified_params = dict(modified_model.named_parameters())
    
    changed_count = 0
    total_count = 0
    max_diff = 0.0
    
    # Get base model parameter names (excluding any LoRA-specific params if present)
    base_param_names = [name for name in base_params.keys() if 'lora_' not in name.lower()]
    
    # Sample parameters to check - focus on MLP layers where LoRA is typically applied
    if sample_layers:
        # Prioritize checking MLP layer parameters
        mlp_params = [name for name in base_param_names if 'mlp' in name.lower()]
        other_params = [name for name in base_param_names if 'mlp' not in name.lower()]
        param_names_to_check = (mlp_params[:sample_layers * 3] + other_params[:sample_layers * 3])[:sample_layers * 6]
    else:
        param_names_to_check = base_param_names
    
    # Compare parameters
    for name in param_names_to_check:
        # For LoRA models, the base parameter still exists but may have same name
        # We check if this parameter exists in the modified model
        if name in modified_params:
            total_count += 1
            try:
                # Get parameter values
                base_val = base_params[name].detach()
                mod_val = modified_params[name].detach()
                
                # Check if shapes match (they should for base params)
                if base_val.shape == mod_val.shape:
                    diff = torch.max(torch.abs(base_val - mod_val)).item()
                    if diff > 1e-6:
                        changed_count += 1
                        max_diff = max(max_diff, diff)
            except Exception as e:
                # Skip parameters that can't be compared
                print(f"  [Debug] Could not compare {name}: {e}")
                total_count -= 1
                continue
    
    return changed_count, total_count, max_diff


def main(args):
    print("\n" + "="*80)
    print("PRAG INFERENCE WITH COMPREHENSIVE VERIFICATION")
    print("="*80)
    print(f"Inference Method: {args.inference_method.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Sample size: {args.sample if args.sample != -1 else 'ALL'}")
    print("="*80 + "\n")
    
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    
    # Load base model (keep a reference for verification)
    print("\n[STEP 1] Loading base model (without LoRA adapters)...")
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    print(f"✓ Base model loaded: {args.model_name}")
    
    # Save base model state for comparison (only for first sample in debug mode)
    base_model_for_comparison = None
    if args.inference_method != "icl":
        print("  → Saving base model state for later comparison...")
    
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
    
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"{'='*80}")
        
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata), 
                                   desc=f"Processing {filename}"):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]
            
            # Show detailed info for first few samples
            show_details = (test_id - start_with) < 3

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
                # ICL mode: Use base model with passages in prompt
                if show_details:
                    print(f"\n--- Sample {test_id} (ICL Mode) ---")
                    print(f"Question: {question[:100]}...")
                    print(f"Using: Base model + {len(passages)} passages in prompt")
                ret.append(get_pred(model, psgs=passages))
            else:
                # PRAG or COMBINE mode: Load and merge LoRA adapters
                if show_details:
                    print(f"\n{'='*80}")
                    print(f"Sample {test_id} - {args.inference_method.upper()} Mode")
                    print(f"{'='*80}")
                    print(f"Question: {question[:100]}...")
                    print(f"Answer: {answer}")
                    print(f"Number of passages: {len(passages)}")
                
                # Step 1: Load adapters
                if show_details:
                    print(f"\n[STEP 2] Loading LoRA adapters for {len(passages)} passages...")
                
                adapter_paths_loaded = []
                for pid in range(len(passages)):
                    adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
                    
                    if show_details:
                        print(f"\n  Passage {pid}:")
                        print(f"    Path: {adapter_path}")
                        print(f"    Exists: {os.path.exists(adapter_path)}")
                        if os.path.exists(adapter_path):
                            files = os.listdir(adapter_path)
                            print(f"    Files: {files}")
                            safetensor_file = os.path.join(adapter_path, "adapter_model.safetensors")
                            if os.path.exists(safetensor_file):
                                size = os.path.getsize(safetensor_file)
                                print(f"    adapter_model.safetensors size: {size:,} bytes ({size/1024:.2f} KB)")
                    
                    if pid == 0:
                        # Load first adapter without specifying adapter_name
                        # This matches how it was saved during encoding
                        model = PeftModel.from_pretrained(
                            model, 
                            adapter_path,
                            is_trainable = False
                        )
                        # Get the default adapter name that was assigned
                        first_adapter_name = list(model.peft_config.keys())[0] if hasattr(model, 'peft_config') else "default"
                        adapter_paths_loaded.append((pid, adapter_path, first_adapter_name))
                        if show_details:
                            print(f"    ✓ Loaded as adapter: '{first_adapter_name}'")
                    else:
                        # Load additional adapters with unique names
                        adapter_name = f"adapter_{pid}"
                        model.load_adapter(adapter_path, adapter_name=adapter_name)
                        adapter_paths_loaded.append((pid, adapter_path, adapter_name))
                        if show_details:
                            print(f"    ✓ Loaded as adapter: '{adapter_name}'")
                
                # Step 2: Get list of all loaded adapter names
                if hasattr(model, 'peft_config'):
                    adapter_names = list(model.peft_config.keys())
                else:
                    # Fallback: assume default naming
                    adapter_names = [first_adapter_name] + [f"adapter_{i}" for i in range(1, len(passages))]
                
                if show_details:
                    print(f"\n[STEP 3] All loaded adapters:")
                    for i, name in enumerate(adapter_names[:len(passages)]):
                        print(f"    {i}. '{name}'")
                
                # Step 3: Merge adapters
                if show_details:
                    print(f"\n[STEP 4] Merging {len(passages)} adapters into single 'merge' adapter...")
                    print(f"  Merge method: linear")
                    print(f"  Weights: {[1.0/len(passages)] * len(passages)} (equal contribution)")
                
                num_adapters = len(passages)
                try:
                    model.add_weighted_adapter(
                        adapters = adapter_names[:num_adapters], 
                        weights = [1.0 / num_adapters] * num_adapters,
                        adapter_name = "merge", 
                        combination_type = "linear",
                    )
                    if show_details:
                        print(f"  ✓ Merge successful - created adapter 'merge'")
                except Exception as e:
                    if show_details:
                        print(f"  ❌ Merge failed: {e}")
                    raise
                
                # Step 4: Set merged adapter as active
                model.set_adapter("merge")
                
                # Step 5: Verify adapter is active
                if show_details:
                    print(f"\n[STEP 5] Verifying merged adapter is active...")
                    if hasattr(model, 'active_adapter'):
                        current_adapter = model.active_adapter
                        if current_adapter == "merge":
                            print(f"  ✓ Active adapter: '{current_adapter}' (correct)")
                        else:
                            print(f"  ❌ Active adapter: '{current_adapter}' (expected 'merge')")
                            warnings.warn(f"Expected adapter 'merge' but got '{current_adapter}'")
                    else:
                        print(f"  ⚠ Cannot verify active adapter (attribute not available)")
                
                # Step 6: Compare with base model (for first sample only)
                if show_details and (test_id - start_with) == 0:
                    print(f"\n[STEP 6] Comparing merged model with base model...")
                    # Load a clean base model for comparison
                    try:
                        base_model_temp, _, _ = get_model(args.model_name, max_new_tokens=args.max_new_tokens)
                        changed, total, max_diff = compare_model_parameters(base_model_temp, model, sample_layers=3)
                        del base_model_temp
                        torch.cuda.empty_cache()
                        
                        if total == 0:
                            print(f"  ⚠ Warning: No comparable parameters found (0/0)")
                            print(f"    This may indicate a model structure issue")
                            print(f"    LoRA adapters may still be active but comparison failed")
                        elif changed > 0:
                            print(f"  ✓ Model parameters CHANGED by LoRA")
                            print(f"    - {changed}/{total} sampled parameters are different")
                            print(f"    - Maximum difference: {max_diff:.6f}")
                            print(f"  ✅ Conclusion: Using LoRA-modified model")
                        else:
                            print(f"  ❌ Model parameters UNCHANGED")
                            print(f"    - {changed}/{total} sampled parameters are different")
                            print(f"  ⚠ Warning: LoRA may not be active!")
                    except Exception as e:
                        print(f"  ⚠ Could not compare models: {e}")
                
                # Step 7: Generate prediction
                if show_details:
                    print(f"\n[STEP 7] Generating prediction...")
                    if args.inference_method == "prag":
                        print(f"  Mode: PRAG (no passages in prompt, knowledge in LoRA)")
                    else:
                        print(f"  Mode: COMBINE (passages in prompt + LoRA knowledge)")
                
                pred = get_pred(model, psgs=None if args.inference_method == "prag" else passages)
                ret.append(pred)
                
                if show_details:
                    print(f"  Generated answer: {pred['text']}")
                    print(f"  Ground truth: {answer}")
                    print(f"  F1 Score: {pred['f1']:.4f}")
                    print(f"  EM Score: {pred['em']:.4f}")
                
                # Cleanup
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        # Save predictions
        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)
        
        print(f"\n{'='*80}")
        print(f"FINAL EVALUATION - {filename}")
        print(f"{'='*80}")

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        
        print(f"\nResults on {len(ret)} samples:")
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
            print(f"  {met.upper()}: {acc:.4f}")
        
        # Verify metric calculation
        print(f"\n[METRIC VERIFICATION]")
        print(f"  Total samples: {len(ret)}")
        print(f"  Sample metric check (first 3 samples):")
        for i, item in enumerate(ret[:3]):
            print(f"    Sample {i}: F1={item.get('f1', 'N/A')}, EM={item.get('em', 'N/A')}, "
                  f"Pred='{item.get('text', '')[:50]}...', GT='{item.get('answer', '')[:50]}...'")
        
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)
        
        print(f"\n✓ Results saved to: {output_dir}")
        print(f"{'='*80}\n")


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
