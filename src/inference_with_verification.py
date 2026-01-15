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


def verify_lora_adapters(model, adapter_name="merge"):
    """Verify that LoRA adapters are actually loaded and active
    
    IMPORTANT: LoRA works by adding adapter layers (lora_A, lora_B), NOT by modifying
    base model weights. Base weights remain FROZEN. This function checks if LoRA
    adapter layers exist and contain non-zero trained weights.
    """
    lora_params = {}
    lora_a_params = []
    lora_b_params = []
    
    # Find all LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' in name.lower():
            lora_params[name] = param
            if 'lora_a' in name.lower():
                lora_a_params.append((name, param))
            elif 'lora_b' in name.lower():
                lora_b_params.append((name, param))
    
    # Check results
    has_lora = len(lora_params) > 0
    has_both_matrices = len(lora_a_params) > 0 and len(lora_b_params) > 0
    
    # Check if parameters are non-zero (trained)
    non_zero_count = 0
    total_lora_params = len(lora_params)
    mean_abs_value = 0.0
    
    if has_lora:
        for name, param in lora_params.items():
            param_val = param.detach()
            abs_mean = torch.abs(param_val).mean().item()
            mean_abs_value += abs_mean
            if abs_mean > 1e-6:
                non_zero_count += 1
        mean_abs_value /= total_lora_params if total_lora_params > 0 else 1
    
    # Check active adapter
    active_adapter = None
    if hasattr(model, 'active_adapter'):
        active_adapter = model.active_adapter
    elif hasattr(model, 'active_adapters'):
        active_adapter = model.active_adapters
    
    return {
        'has_lora': has_lora,
        'lora_a_count': len(lora_a_params),
        'lora_b_count': len(lora_b_params),
        'total_lora_params': total_lora_params,
        'non_zero_params': non_zero_count,
        'mean_abs_value': mean_abs_value,
        'active_adapter': active_adapter,
        'sample_params': list(lora_params.keys())[:5],  # First 5 for display
    }


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
                
                # Step 6: Verify LoRA adapters are loaded (for first sample only)
                if show_details and (test_id - start_with) == 0:
                    print(f"\n[STEP 6] Verifying LoRA adapter layers...")
                    print(f"  NOTE: LoRA keeps base weights FROZEN and adds adapter layers")
                    print(f"  We check if lora_A and lora_B layers exist and contain trained weights")
                    try:
                        lora_info = verify_lora_adapters(model, adapter_name="merge")
                        
                        if lora_info['has_lora']:
                            print(f"  ✓ LoRA adapter layers found:")
                            print(f"    - lora_A parameters: {lora_info['lora_a_count']}")
                            print(f"    - lora_B parameters: {lora_info['lora_b_count']}")
                            print(f"    - Total LoRA params: {lora_info['total_lora_params']}")
                            print(f"    - Non-zero params: {lora_info['non_zero_params']}/{lora_info['total_lora_params']}")
                            print(f"    - Mean |weight|: {lora_info['mean_abs_value']:.6f}")
                            
                            if lora_info['active_adapter']:
                                print(f"  ✓ Active adapter: {lora_info['active_adapter']}")
                            
                            if lora_info['sample_params']:
                                print(f"  Sample LoRA parameters (first 3):")
                                for param_name in lora_info['sample_params'][:3]:
                                    print(f"    - {param_name}")
                            
                            if lora_info['non_zero_params'] > 0:
                                print(f"  ✅ LoRA adapters are LOADED and ACTIVE")
                                print(f"     They will modify model outputs during inference")
                            else:
                                print(f"  ⚠ LoRA layers exist but all weights are zero!")
                                print(f"     This suggests adapters were not trained")
                        else:
                            print(f"  ❌ No LoRA adapter layers found in model!")
                            print(f"     Adapters may not have been loaded correctly")
                    except Exception as e:
                        print(f"  ⚠ Could not verify LoRA adapters: {e}")
                
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
            acc = sum(float(d.get(met, 0)) for d in ret) / len(ret)
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
