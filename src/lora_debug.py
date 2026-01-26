"""
LoRA Debug Utilities for PRAG

This module provides debugging functions to inspect LoRA weights during:
1. Encoding (training) phase - before and after training
2. Inference phase - loading, merging, and comparing with base model
"""

import os
import torch
from typing import Dict, List, Optional

# Try to import safetensors for direct file reading
try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Threshold for considering a tensor as "zero"
ZERO_THRESHOLD = 1e-10


def get_lora_weights(model, adapter_name: str = "default") -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract LoRA A and B weights from a PeftModel.
    
    Returns:
        Dict mapping layer names to {'lora_A': tensor, 'lora_B': tensor}
    """
    lora_weights = {}
    
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            # Parse layer name
            # Format: base_model.model.model.layers.X.mlp.{up_proj|down_proj|gate_proj}.lora_{A|B}.{adapter_name}.weight
            parts = name.split('.')
            layer_key = '.'.join(parts[:-2])  # Remove lora_X.adapter_name.weight
            
            if adapter_name not in name and adapter_name != "all":
                continue
                
            if layer_key not in lora_weights:
                lora_weights[layer_key] = {}
            
            if 'lora_A' in name:
                lora_weights[layer_key]['lora_A'] = param.data.clone()
            elif 'lora_B' in name:
                lora_weights[layer_key]['lora_B'] = param.data.clone()
    
    return lora_weights


def print_lora_weight_summary(lora_weights: Dict[str, Dict[str, torch.Tensor]], 
                               title: str = "LoRA Weights Summary",
                               num_layers: int = 3):
    """
    Print a summary of LoRA weights including shape, mean, std, min, max.
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    
    layers = sorted(lora_weights.keys())
    displayed = 0
    
    for layer_name in layers:
        if num_layers > 0 and displayed >= num_layers:
            print(f"\n... (showing {num_layers} of {len(layers)} layers, set num_layers=-1 to show all)")
            break
            
        weights = lora_weights[layer_name]
        print(f"\n[Layer: {layer_name}]")
        
        for weight_type in ['lora_A', 'lora_B']:
            if weight_type in weights:
                w = weights[weight_type]
                print(f"  {weight_type}:")
                print(f"    Shape: {tuple(w.shape)}")
                print(f"    Mean:  {w.mean().item():.8f}")
                print(f"    Std:   {w.std().item():.8f}")
                print(f"    Min:   {w.min().item():.8f}")
                print(f"    Max:   {w.max().item():.8f}")
                print(f"    First 5 values: {w.flatten()[:5].tolist()}")
        
        displayed += 1


def print_lora_storage_info(save_path: str, title: str = "LoRA Storage Info"):
    """
    Print information about LoRA storage location and files.
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(f"  Path: {save_path}")
    
    if os.path.exists(save_path):
        try:
            files = os.listdir(save_path)
            print(f"  Files: {files}")
            for f in files:
                file_path = os.path.join(save_path, f)
                try:
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f"    - {f}: {size} bytes ({size/1024:.2f} KB)")
                except OSError as e:
                    print(f"    - {f}: [Error getting file info: {e}]")
        except OSError as e:
            print(f"  [ERROR] Cannot list directory: {e}")
    else:
        print("  [WARNING] Path does not exist!")


def compare_lora_weights(weights1: Dict[str, Dict[str, torch.Tensor]], 
                         weights2: Dict[str, Dict[str, torch.Tensor]],
                         title: str = "LoRA Weight Comparison",
                         num_layers: int = 3):
    """
    Compare two sets of LoRA weights to check if they differ.
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    
    layers = sorted(set(weights1.keys()) | set(weights2.keys()))
    displayed = 0
    total_diff = 0
    identical_count = 0
    
    for layer_name in layers:
        if displayed >= num_layers and num_layers > 0:
            continue
            
        w1 = weights1.get(layer_name, {})
        w2 = weights2.get(layer_name, {})
        
        for weight_type in ['lora_A', 'lora_B']:
            if weight_type in w1 and weight_type in w2:
                diff = (w1[weight_type] - w2[weight_type]).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                total_diff += mean_diff
                
                if max_diff < 1e-8:
                    identical_count += 1
                    status = "IDENTICAL"
                else:
                    status = "DIFFERENT"
                
                if displayed < num_layers or num_layers < 0:
                    print(f"\n[{layer_name}.{weight_type}]")
                    print(f"  Status: {status}")
                    print(f"  Max diff:  {max_diff:.8f}")
                    print(f"  Mean diff: {mean_diff:.8f}")
        
        displayed += 1
    
    print(f"\n[Summary]")
    print(f"  Total layers compared: {len(layers)}")
    print(f"  Identical weight pairs: {identical_count}")
    print(f"  Total mean difference: {total_diff:.8f}")


def compare_model_weights_before_after_lora(
    base_model_state_dict: Dict[str, torch.Tensor],
    merged_model_state_dict: Dict[str, torch.Tensor],
    target_modules: List[str] = ['down_proj', 'gate_proj', 'up_proj'],
    num_layers: int = 3
):
    """
    Compare base model weights with LoRA-merged model weights.
    Only compares the modules that LoRA targets.
    """
    print("\n" + "=" * 80)
    print(" Base Model vs LoRA-Merged Model Weight Comparison")
    print("=" * 80)
    
    relevant_keys = []
    for key in base_model_state_dict.keys():
        if any(target in key for target in target_modules):
            # Skip lora-specific parameters
            if 'lora' not in key.lower():
                relevant_keys.append(key)
    
    print(f"\nComparing {len(relevant_keys)} relevant weight tensors (target modules: {target_modules})")
    
    displayed = 0
    total_diff = 0
    identical_count = 0
    
    for key in sorted(relevant_keys):
        if displayed >= num_layers and num_layers > 0:
            continue
            
        if key not in merged_model_state_dict:
            if displayed < num_layers or num_layers < 0:
                print(f"\n[{key}] - NOT FOUND in merged model")
            continue
        
        base_w = base_model_state_dict[key]
        merged_w = merged_model_state_dict[key]
        
        diff = (base_w - merged_w).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        total_diff += mean_diff
        
        if max_diff < 1e-8:
            identical_count += 1
            status = "IDENTICAL (LoRA NOT applied)"
        else:
            status = "DIFFERENT (LoRA applied)"
        
        if displayed < num_layers or num_layers < 0:
            print(f"\n[{key}]")
            print(f"  Status: {status}")
            print(f"  Max diff:  {max_diff:.8f}")
            print(f"  Mean diff: {mean_diff:.8f}")
            print(f"  Base mean:   {base_w.mean().item():.8f}")
            print(f"  Merged mean: {merged_w.mean().item():.8f}")
        
        displayed += 1
    
    print(f"\n[Summary]")
    print(f"  Total relevant layers: {len(relevant_keys)}")
    print(f"  Identical (LoRA NOT applied): {identical_count}")
    print(f"  Different (LoRA applied): {len(relevant_keys) - identical_count}")
    print(f"  Total mean difference across all layers: {total_diff:.8f}")
    
    if identical_count == len(relevant_keys):
        print("\n  [WARNING] All weights are identical - LoRA may not be properly applied!")


def print_adapter_info(model, title: str = "Adapter Information"):
    """
    Print information about loaded adapters in a PeftModel.
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    
    if hasattr(model, 'peft_config'):
        print(f"  PEFT Config available: Yes")
        for adapter_name, config in model.peft_config.items():
            print(f"\n  [Adapter: {adapter_name}]")
            print(f"    - r (rank): {config.r}")
            print(f"    - lora_alpha: {config.lora_alpha}")
            print(f"    - target_modules: {config.target_modules}")
            print(f"    - lora_dropout: {config.lora_dropout}")
    else:
        print(f"  PEFT Config available: No (not a PeftModel)")
    
    if hasattr(model, 'active_adapter'):
        print(f"\n  Active adapter: {model.active_adapter}")
    
    if hasattr(model, 'active_adapters'):
        print(f"  Active adapters: {model.active_adapters}")


def get_merged_lora_weights_info(model, adapter_names: List[str], 
                                  weights: List[float],
                                  title: str = "Merged LoRA Weights Preview"):
    """
    Show what the merged LoRA weights would look like after add_weighted_adapter.
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(f"  Adapters to merge: {adapter_names}")
    print(f"  Weights: {weights}")
    print(f"  Combination type: cat (concatenation)")
    
    # Get weights for each adapter
    for adapter_name in adapter_names:
        weights_dict = get_lora_weights(model, adapter_name)
        if weights_dict:
            print(f"\n  Adapter '{adapter_name}' has {len(weights_dict)} layer groups")
            # Show first layer
            first_layer = sorted(weights_dict.keys())[0]
            print(f"    First layer: {first_layer}")
            if 'lora_A' in weights_dict[first_layer]:
                print(f"      lora_A shape: {tuple(weights_dict[first_layer]['lora_A'].shape)}")
            if 'lora_B' in weights_dict[first_layer]:
                print(f"      lora_B shape: {tuple(weights_dict[first_layer]['lora_B'].shape)}")


def debug_encode_before_training(model, init_adapter_path: str):
    """
    Debug function to call before training in encode.py
    """
    print("\n" + "#" * 80)
    print(" DEBUG: ENCODE - BEFORE TRAINING")
    print("#" * 80)
    
    print_lora_storage_info(init_adapter_path, "Initial LoRA (base_weight) Storage Info")
    
    # Get initial LoRA weights
    lora_weights = get_lora_weights(model, "default")
    print_lora_weight_summary(lora_weights, "Initial LoRA Weights (from base_weight)", num_layers=3)
    
    return lora_weights


def debug_encode_after_training(model, save_path: str, initial_weights: Dict[str, Dict[str, torch.Tensor]]):
    """
    Debug function to call after training in encode.py
    """
    print("\n" + "#" * 80)
    print(" DEBUG: ENCODE - AFTER TRAINING")
    print("#" * 80)
    
    # Get trained LoRA weights
    trained_weights = get_lora_weights(model, "default")
    print_lora_weight_summary(trained_weights, "Trained LoRA Weights", num_layers=3)
    
    # Compare before and after
    compare_lora_weights(initial_weights, trained_weights, 
                        "Comparing Initial vs Trained LoRA Weights", num_layers=3)
    
    print_lora_storage_info(save_path, "Trained LoRA Storage Info (Save Location)")


def debug_inference_load_adapter(model, adapter_path: str, adapter_name: str, pid: int):
    """
    Debug function to call after loading an adapter in inference.py
    """
    print("\n" + "#" * 80)
    print(f" DEBUG: INFERENCE - LOADED ADAPTER (pid={pid})")
    print("#" * 80)
    
    print_lora_storage_info(adapter_path, f"Loaded LoRA Storage Info (Adapter '{adapter_name}')")
    
    # Directly read the safetensors file to see what's actually stored
    print("\n  [Direct file read - what's actually saved on disk:]")
    read_safetensors_file(adapter_path, f"Safetensors File for Adapter '{adapter_name}'")
    
    print_adapter_info(model, f"Adapter Info after loading '{adapter_name}'")
    
    # Get LoRA weights for this adapter (what's loaded in memory)
    print("\n  [Model memory - what's loaded into the model:]")
    lora_weights = get_lora_weights(model, adapter_name)
    print_lora_weight_summary(lora_weights, f"LoRA Weights for Adapter '{adapter_name}' (in memory)", num_layers=2)


def debug_inference_after_merge(model, adapter_names: List[str], 
                                 base_model_state_dict: Optional[Dict[str, torch.Tensor]] = None):
    """
    Debug function to call after merging adapters in inference.py
    """
    print("\n" + "#" * 80)
    print(" DEBUG: INFERENCE - AFTER MERGING ADAPTERS")
    print("#" * 80)
    
    print(f"  Merged adapters: {adapter_names}")
    print(f"  Active adapter after merge: {model.active_adapter if hasattr(model, 'active_adapter') else 'N/A'}")
    
    # Get merged LoRA weights
    merged_weights = get_lora_weights(model, "merge")
    print_lora_weight_summary(merged_weights, "Merged LoRA Weights (adapter='merge')", num_layers=3)
    
    # Compare with base model if provided
    if base_model_state_dict is not None:
        # Get current model state dict (after merge)
        current_state_dict = {k: v.clone() for k, v in model.state_dict().items() 
                              if 'lora' not in k.lower()}
        compare_model_weights_before_after_lora(
            base_model_state_dict, 
            current_state_dict,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            num_layers=3
        )


def capture_base_model_weights(model, target_modules: List[str] = ['down_proj', 'gate_proj', 'up_proj']) -> Dict[str, torch.Tensor]:
    """
    Capture base model weights before LoRA is applied.
    """
    weights = {}
    for name, param in model.named_parameters():
        if any(target in name for target in target_modules):
            if 'lora' not in name.lower():
                weights[name] = param.data.clone()
    return weights


def read_safetensors_file(adapter_path: str, title: str = "Safetensors File Contents"):
    """
    Directly read and display the contents of a safetensors file to verify what's actually saved.
    This is useful for debugging when model loading seems to not work correctly.
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    
    if not os.path.exists(safetensors_path):
        print(f"  [ERROR] File not found: {safetensors_path}")
        return None
    
    if not SAFETENSORS_AVAILABLE:
        print("  [ERROR] safetensors library not installed. Run: pip install safetensors")
        return None
    
    try:
        weights = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            print(f"  File: {safetensors_path}")
            print(f"  Keys in file: {list(f.keys())}")
            
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = tensor
                
                # Show summary for lora_A and lora_B
                if 'lora_A' in key or 'lora_B' in key:
                    print(f"\n  [{key}]")
                    print(f"    Shape: {tuple(tensor.shape)}")
                    print(f"    Mean:  {tensor.mean().item():.8f}")
                    print(f"    Std:   {tensor.std().item():.8f}")
                    print(f"    Min:   {tensor.min().item():.8f}")
                    print(f"    Max:   {tensor.max().item():.8f}")
                    print(f"    First 5 values: {tensor.flatten()[:5].tolist()}")
                    
                    # Check if all zeros
                    if tensor.abs().max().item() < ZERO_THRESHOLD:
                        print(f"    [WARNING] This tensor is all zeros!")
        
        return weights
    except Exception as e:
        print(f"  [ERROR] Failed to read safetensors file: {e}")
        return None


def verify_saved_adapter(save_path: str):
    """
    Verify that the adapter was saved correctly by reading the safetensors file directly.
    This should be called after model.save_pretrained() to confirm the weights are saved.
    """
    print("\n" + "#" * 80)
    print(" VERIFICATION: Checking saved adapter file")
    print("#" * 80)
    
    read_safetensors_file(save_path, f"Saved Adapter Contents at {save_path}")
