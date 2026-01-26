import os
import re
import json
import torch
import string
import numpy as np
from collections import Counter
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import safetensors.torch as safetensors_torch
except ImportError:
    safetensors_torch = None

from root_dir_path import ROOT_DIR
from prompt_template import get_prompt

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data_aug")

# Debugging constants
DEBUG_MAX_ITEMS = 2  # Maximum number of items to show detailed debug output for

class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


def load_data(data_name, data_type, model_name):
    solve_dataset = []
    input_dir = os.path.join(DATA_ROOT_DIR, data_name, model_name)
    files = [f for f in os.listdir(input_dir)]


    if len(files) > 1: # more types in dataset
        if data_type == "total": # merge all types to total
            all_data = {}
            for filename in files:
                with open(os.path.join(input_dir, filename), "r") as fin:
                    all_data[filename] = json.load(fin)
            total_data = []
            idx = {filename: 0 for filename in files}
            for data in all_data["total.json"]:
                typ = data["type"] + ".json"
                if idx[typ] == len(all_data[typ]):
                    break 
                aim_data = all_data[typ][idx[typ]]
                assert aim_data["question"] == data["question"]
                idx[typ] += 1
                total_data.append(aim_data)
            return [["total.json", total_data]]
        for filename in files:
            if filename != "total.json":
                with open(os.path.join(input_dir, filename), "r") as fin:
                    solve_dataset.append((filename, json.load(fin)))
        if data_type is None:
            return solve_dataset
        else:
            data_type = data_type + ".json"
            if data_type not in [v[0] for v in solve_dataset]:
                raise ValueError(f"Invalid {data_type} in Dataset {data_name}")
            tmp = []
            for filename, dataset in solve_dataset:
                if filename == data_type:
                    tmp.append((filename, dataset))
            return tmp
    else:
        with open(os.path.join(input_dir, "total.json"), "r") as fin:
            solve_dataset.append(("total.json", json.load(fin)))
        return solve_dataset
    

def get_model_path(model_name):
    if model_name == "llama3-8b-instruct": 
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5-1.5b-instruct":
        return "Qwen/Qwen2.5-1.5B-Instruct"
    elif model_name == "llama3.2-1b-instruct":
        return "meta-llama/Llama-3.2-1B-Instruct"
    else:
        return model_name


def get_model(model_name, max_new_tokens=20):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generation_config = dict(
        num_beams=1, 
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    return model, tokenizer, generation_config

# -------------------------------- for augmentation ----------------------------------------

def model_generate(prompt, model, tokenizer, generation_config):
    messages = [{
        'role': 'user', 
        'content': prompt,
    }]
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True
    )
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    output = model.generate(
        input_ids, 
        attention_mask = torch.ones(input_ids.shape).to(model.device),
        **generation_config
    )
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text

# ------------------------------------------------------------------------------------

def read_complete(filepath):
    try:
        with open(filepath, "r") as fin:
            data = json.load(fin)
        return data, len(data)
    except:
        return [], 0

    
def evaluate(pred, ground_truth, with_cot=False):
    if not with_cot:
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else:
        if "the answer is" in pred:
            pred = pred[pred.find("the answer is") + len("the answer is"):]
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip() 

    em = BaseDataset.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def predict(model, tokenizer, generation_config, question, with_cot, passages = None):
    model.eval()
    input_ids = get_prompt(
        tokenizer, 
        question, 
        passages = passages, 
        with_cot = with_cot)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask = torch.ones(input_ids.shape).to(model.device),
            **generation_config)
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text


# -------------------------------- LoRA debugging utilities ----------------------------------------

def print_lora_weights(model, tag="", max_layers=3):
    """
    Print LoRA A and B matrix statistics for debugging.
    
    Args:
        model: The PeftModel instance
        tag: A tag to identify when this is called (e.g., "after_training", "after_loading")
        max_layers: Maximum number of layers to print detailed info for
    """
    print(f"\n{'='*60}")
    print(f"LoRA Weight Debug Info [{tag}]")
    print(f"{'='*60}")
    
    try:
        # Get the state dict of the adapter
        if hasattr(model, 'peft_config'):
            print(f"PEFT Config: {model.peft_config}")
        
        if hasattr(model, 'active_adapter'):
            print(f"Active Adapter: {model.active_adapter}")
        
        # Iterate through model parameters to find LoRA weights
        lora_params = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_params[name] = param.data.clone()
        
        if not lora_params:
            print("WARNING: No LoRA parameters found in the model!")
            return
        
        print(f"\nFound {len(lora_params)} LoRA parameters:")
        
        # Group by layer
        layer_stats = {}
        for name, data in lora_params.items():
            # Extract layer info
            parts = name.split('.')
            layer_key = '.'.join([p for p in parts if 'layer' in p or 'layers' in p][:2])
            if not layer_key:
                layer_key = 'other'
            
            if layer_key not in layer_stats:
                layer_stats[layer_key] = []
            
            stats = {
                'name': name,
                'shape': tuple(data.shape),
                'mean': data.float().mean().item(),
                'std': data.float().std().item(),
                'min': data.float().min().item(),
                'max': data.float().max().item(),
                'abs_mean': data.float().abs().mean().item(),
                'norm': data.float().norm().item(),
            }
            layer_stats[layer_key].append(stats)
        
        # Print summary for each layer
        printed_layers = 0
        for layer_key in sorted(layer_stats.keys()):
            if printed_layers >= max_layers and max_layers > 0:
                remaining = len(layer_stats) - printed_layers
                print(f"\n... and {remaining} more layers (use max_layers=-1 to see all)")
                break
            
            print(f"\n--- Layer: {layer_key} ---")
            for stats in layer_stats[layer_key]:
                print(f"  {stats['name']}")
                print(f"    Shape: {stats['shape']}")
                print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
                print(f"    Abs Mean: {stats['abs_mean']:.6f}, Norm: {stats['norm']:.6f}")
            printed_layers += 1
        
        # Print overall statistics
        all_means = [s['abs_mean'] for stats_list in layer_stats.values() for s in stats_list]
        all_norms = [s['norm'] for stats_list in layer_stats.values() for s in stats_list]
        print(f"\n--- Overall Statistics ---")
        print(f"Total LoRA parameters: {len(lora_params)}")
        print(f"Average absolute mean across all params: {np.mean(all_means):.6f}")
        print(f"Average norm across all params: {np.mean(all_norms):.6f}")
        
    except Exception as e:
        print(f"Error while printing LoRA weights: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")


def print_adapter_weights_from_path(adapter_path, tag=""):
    """
    Print LoRA weight statistics from a saved adapter path.
    
    Args:
        adapter_path: Path to the saved adapter directory
        tag: A tag to identify when this is called
    """
    print(f"\n{'='*60}")
    print(f"LoRA Weight Debug Info from Path [{tag}]")
    print(f"Path: {adapter_path}")
    print(f"{'='*60}")
    
    try:
        # Try to load from safetensors first, then bin
        safetensor_path = os.path.join(adapter_path, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_path, "adapter_model.bin")
        
        if os.path.exists(safetensor_path):
            if safetensors_torch is None:
                print("WARNING: safetensors not installed, cannot load .safetensors files")
                return
            state_dict = safetensors_torch.load_file(safetensor_path)
            print(f"Loaded from: adapter_model.safetensors")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location='cpu')
            print(f"Loaded from: adapter_model.bin")
        else:
            print(f"WARNING: No adapter model file found at {adapter_path}")
            return
        
        print(f"\nFound {len(state_dict)} parameters:")
        
        # Print stats for each parameter
        lora_a_stats = []
        lora_b_stats = []
        
        for name, data in sorted(state_dict.items()):
            stats = {
                'name': name,
                'shape': tuple(data.shape),
                'mean': data.float().mean().item(),
                'std': data.float().std().item(),
                'min': data.float().min().item(),
                'max': data.float().max().item(),
                'abs_mean': data.float().abs().mean().item(),
                'norm': data.float().norm().item(),
            }
            
            if 'lora_A' in name:
                lora_a_stats.append(stats)
            elif 'lora_B' in name:
                lora_b_stats.append(stats)
            
            print(f"\n{name}")
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
            print(f"  Abs Mean: {stats['abs_mean']:.6f}, Norm: {stats['norm']:.6f}")
        
        # Summary
        if lora_a_stats:
            avg_a_norm = np.mean([s['norm'] for s in lora_a_stats])
            print(f"\n--- LoRA_A Summary ---")
            print(f"Count: {len(lora_a_stats)}, Avg Norm: {avg_a_norm:.6f}")
        
        if lora_b_stats:
            avg_b_norm = np.mean([s['norm'] for s in lora_b_stats])
            print(f"\n--- LoRA_B Summary ---")
            print(f"Count: {len(lora_b_stats)}, Avg Norm: {avg_b_norm:.6f}")
            # Note: LoRA_B is typically initialized to zeros, so non-zero values indicate training
            if avg_b_norm < 1e-8:
                print("WARNING: LoRA_B weights appear to be all zeros (not trained)!")
            else:
                print(f"LoRA_B has non-zero weights (training occurred)")
        
    except Exception as e:
        print(f"Error while loading adapter weights: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")


def compare_lora_weights(model1_or_path, model2_or_path, tag1="model1", tag2="model2"):
    """
    Compare LoRA weights between two models or paths to check if they are different.
    
    Returns True if weights are significantly different, False otherwise.
    """
    def normalize_key(key):
        """
        Normalize LoRA key names for comparison.
        Keys from saved adapters: base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
        Keys from PeftModel: base_model.model.base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight
        
        This function extracts the essential part: layers.X.mlp.{proj}.lora_{A/B}
        """
        # Remove common prefixes
        key = key.replace('base_model.model.base_model.model.model.', '')
        key = key.replace('base_model.model.model.', '')
        key = key.replace('base_model.model.', '')
        # Remove .default suffix (present in PeftModel but not in saved files)
        key = key.replace('.default.weight', '.weight')
        return key
    
    def get_state_dict(model_or_path):
        if isinstance(model_or_path, str):
            # It's a path
            safetensor_path = os.path.join(model_or_path, "adapter_model.safetensors")
            bin_path = os.path.join(model_or_path, "adapter_model.bin")
            
            if os.path.exists(safetensor_path):
                if safetensors_torch is None:
                    raise ImportError("safetensors not installed")
                return safetensors_torch.load_file(safetensor_path)
            elif os.path.exists(bin_path):
                return torch.load(bin_path, map_location='cpu')
            else:
                raise FileNotFoundError(f"No adapter model file found at {model_or_path}")
        else:
            # It's a model, extract LoRA params
            params = {}
            for name, param in model_or_path.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    params[name] = param.data.clone()
            return params
    
    print(f"\n{'='*60}")
    print(f"Comparing LoRA weights: {tag1} vs {tag2}")
    print(f"{'='*60}")
    
    try:
        state_dict1 = get_state_dict(model1_or_path)
        state_dict2 = get_state_dict(model2_or_path)
        
        # Normalize keys for comparison
        normalized1 = {normalize_key(k): (k, v) for k, v in state_dict1.items()}
        normalized2 = {normalize_key(k): (k, v) for k, v in state_dict2.items()}
        
        # Find common normalized keys
        norm_keys1 = set(normalized1.keys())
        norm_keys2 = set(normalized2.keys())
        common_norm_keys = norm_keys1 & norm_keys2
        
        if not common_norm_keys:
            print("WARNING: No common keys found between the two models!")
            print(f"Keys in {tag1} (first 5): {list(state_dict1.keys())[:5]}")
            print(f"Keys in {tag2} (first 5): {list(state_dict2.keys())[:5]}")
            print(f"Normalized keys in {tag1} (first 5): {list(norm_keys1)[:5]}")
            print(f"Normalized keys in {tag2} (first 5): {list(norm_keys2)[:5]}")
            return False
        
        print(f"Common parameters (after key normalization): {len(common_norm_keys)}")
        
        total_diff = 0
        significantly_different = False
        
        for norm_key in sorted(common_norm_keys):
            orig_key1, data1 = normalized1[norm_key]
            orig_key2, data2 = normalized2[norm_key]
            
            data1 = data1.float()
            data2 = data2.float()
            
            diff = (data1 - data2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            if max_diff > 1e-6:
                significantly_different = True
                print(f"\n{norm_key}")
                print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
            
            total_diff += mean_diff
        
        avg_diff = total_diff / len(common_norm_keys)
        print(f"\n--- Summary ---")
        print(f"Average difference: {avg_diff:.6f}")
        print(f"Weights are {'DIFFERENT' if significantly_different else 'IDENTICAL (or nearly identical)'}")
        
        return significantly_different
        
    except Exception as e:
        print(f"Error while comparing weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"{'='*60}\n")


def print_lora_sample_values(model_or_path, tag="", num_samples=5, layer_idx=0):
    """
    Print actual sample values from LoRA A and B matrices for detailed inspection.
    
    Args:
        model_or_path: PeftModel instance or path to saved adapter
        tag: A tag to identify when this is called
        num_samples: Number of sample values to print from each matrix
        layer_idx: Which layer to show detailed values for
    """
    print(f"\n{'='*60}")
    print(f"LoRA Sample Values [{tag}]")
    print(f"{'='*60}")
    
    try:
        if isinstance(model_or_path, str):
            # Load from path
            safetensor_path = os.path.join(model_or_path, "adapter_model.safetensors")
            bin_path = os.path.join(model_or_path, "adapter_model.bin")
            
            if os.path.exists(safetensor_path):
                if safetensors_torch is None:
                    print("WARNING: safetensors not installed")
                    return
                state_dict = safetensors_torch.load_file(safetensor_path)
                print(f"Loaded from: {safetensor_path}")
            elif os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location='cpu')
                print(f"Loaded from: {bin_path}")
            else:
                print(f"WARNING: No adapter file found at {model_or_path}")
                return
        else:
            # Extract from model
            state_dict = {}
            for name, param in model_or_path.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    state_dict[name] = param.data.clone()
            print(f"Extracted from model in memory")
        
        # Find LoRA A and B for the specified layer
        target_layer = f"layers.{layer_idx}"
        for name, data in sorted(state_dict.items()):
            if target_layer in name or (layer_idx == 0 and 'layers.0' not in name and 'layer' not in name):
                data = data.float()
                print(f"\n{name}")
                print(f"  Shape: {tuple(data.shape)}")
                print(f"  Stats - Mean: {data.mean().item():.8f}, Std: {data.std().item():.8f}")
                print(f"  Stats - Min: {data.min().item():.8f}, Max: {data.max().item():.8f}")
                print(f"  Stats - Norm: {data.norm().item():.8f}")
                
                # Print sample values
                flat = data.flatten()
                sample_indices = [0, len(flat)//4, len(flat)//2, 3*len(flat)//4, len(flat)-1][:num_samples]
                sample_values = [flat[i].item() for i in sample_indices]
                print(f"  Sample values at indices {sample_indices}:")
                print(f"    {[f'{v:.8f}' for v in sample_values]}")
                
                # Check if all zeros
                if data.abs().max().item() < 1e-10:
                    print(f"  *** ALL ZEROS (not trained) ***")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")


def compare_base_model_with_lora(base_model, lora_model, tag="", max_layers=2):
    """
    Compare base model weights with model that has LoRA applied.
    Shows the difference in weights for LoRA-affected layers only.
    
    Args:
        base_model: The original model without LoRA
        lora_model: The model with LoRA adapters merged/applied
        tag: A tag to identify this comparison
        max_layers: Maximum number of layers to show detailed comparison
    """
    print(f"\n{'='*60}")
    print(f"Base Model vs LoRA Model Comparison [{tag}]")
    print(f"{'='*60}")
    
    try:
        # Get target modules that LoRA affects
        target_modules = ['down_proj', 'gate_proj', 'up_proj']
        
        base_params = dict(base_model.named_parameters())
        lora_params = dict(lora_model.named_parameters())
        
        layers_shown = 0
        total_diff = 0
        num_compared = 0
        
        for name in sorted(base_params.keys()):
            # Check if this is a target module
            is_target = any(tm in name for tm in target_modules)
            if not is_target:
                continue
            
            # Skip if not in lora_model
            if name not in lora_params:
                continue
            
            base_data = base_params[name].data.float()
            lora_data = lora_params[name].data.float()
            
            diff = (base_data - lora_data).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            total_diff += mean_diff
            num_compared += 1
            
            # Only print details for first few layers
            if 'layers.0.' in name or 'layers.1.' in name:
                if layers_shown < max_layers * 3:  # 3 target modules per layer
                    print(f"\n{name}")
                    print(f"  Base shape: {tuple(base_data.shape)}")
                    print(f"  Max diff: {max_diff:.8f}, Mean diff: {mean_diff:.8f}")
                    
                    # Show sample values comparison
                    flat_base = base_data.flatten()
                    flat_lora = lora_data.flatten()
                    sample_idx = [0, len(flat_base)//2, len(flat_base)-1]
                    print(f"  Sample comparison at indices {sample_idx}:")
                    for idx in sample_idx:
                        b_val = flat_base[idx].item()
                        l_val = flat_lora[idx].item()
                        print(f"    [{idx}] Base: {b_val:.8f}, LoRA: {l_val:.8f}, Diff: {abs(b_val-l_val):.8f}")
                    
                    layers_shown += 1
        
        avg_diff = total_diff / num_compared if num_compared > 0 else 0
        print(f"\n--- Summary ---")
        print(f"Compared {num_compared} target module parameters")
        print(f"Average difference: {avg_diff:.8f}")
        if avg_diff < 1e-8:
            print("WARNING: No difference detected! LoRA may not be properly applied.")
        else:
            print("LoRA modifications detected in target modules.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")


def print_merged_lora_info(model, adapter_names, tag=""):
    """
    Print information about merged LoRA adapters.
    
    Args:
        model: PeftModel with multiple adapters loaded
        adapter_names: List of adapter names that were merged
        tag: A tag to identify this output
    """
    print(f"\n{'='*60}")
    print(f"Merged LoRA Info [{tag}]")
    print(f"{'='*60}")
    
    try:
        print(f"Merged adapters: {adapter_names}")
        print(f"Active adapter: {model.active_adapter if hasattr(model, 'active_adapter') else 'N/A'}")
        
        # Count and summarize LoRA parameters
        lora_a_count = 0
        lora_b_count = 0
        lora_a_norms = []
        lora_b_norms = []
        
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                lora_a_count += 1
                lora_a_norms.append(param.data.float().norm().item())
            elif 'lora_B' in name:
                lora_b_count += 1
                lora_b_norms.append(param.data.float().norm().item())
        
        print(f"\nLoRA_A parameters: {lora_a_count}")
        if lora_a_norms:
            print(f"  Avg Norm: {np.mean(lora_a_norms):.6f}")
            print(f"  Min Norm: {min(lora_a_norms):.6f}, Max Norm: {max(lora_a_norms):.6f}")
        
        print(f"\nLoRA_B parameters: {lora_b_count}")
        if lora_b_norms:
            print(f"  Avg Norm: {np.mean(lora_b_norms):.6f}")
            print(f"  Min Norm: {min(lora_b_norms):.6f}, Max Norm: {max(lora_b_norms):.6f}")
            if np.mean(lora_b_norms) < 1e-8:
                print("  WARNING: LoRA_B norms are near zero!")
        
        # Show sample from first layer
        print("\n--- Sample from layers.0 ---")
        for name, param in model.named_parameters():
            if 'layers.0.' in name and ('lora_A' in name or 'lora_B' in name):
                if 'down_proj' in name:
                    data = param.data.float()
                    print(f"  {name}")
                    print(f"    Shape: {tuple(data.shape)}, Norm: {data.norm().item():.6f}")
                    flat = data.flatten()
                    print(f"    First 3 values: {[f'{flat[i].item():.8f}' for i in range(min(3, len(flat)))]}")
                    break
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")