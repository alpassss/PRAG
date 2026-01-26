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
        
        # Find common keys
        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())
        common_keys = keys1 & keys2
        
        if not common_keys:
            print("WARNING: No common keys found between the two models!")
            return False
        
        print(f"Common parameters: {len(common_keys)}")
        
        total_diff = 0
        significantly_different = False
        
        for key in sorted(common_keys):
            data1 = state_dict1[key].float()
            data2 = state_dict2[key].float()
            
            diff = (data1 - data2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            if max_diff > 1e-6:
                significantly_different = True
                print(f"\n{key}")
                print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
            
            total_diff += mean_diff
        
        avg_diff = total_diff / len(common_keys)
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