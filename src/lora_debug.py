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
                               num_layers: int = 3,
                               verbose: bool = False):
    """
    Print a summary of LoRA weights including shape, mean, std, min, max.
    
    Args:
        lora_weights: Dictionary of LoRA weights
        title: Title for the output
        num_layers: Number of layers to display details for (only used if verbose=True)
        verbose: If True, show detailed statistics; if False, only show summary
    """
    if not verbose:
        # Only show basic summary
        num_layers_total = len(lora_weights)
        if num_layers_total > 0:
            # Check for zero weights (potential issues)
            zero_count = 0
            for weights in lora_weights.values():
                for weight_type in ['lora_A', 'lora_B']:
                    if weight_type in weights:
                        if weights[weight_type].abs().max().item() < ZERO_THRESHOLD:
                            zero_count += 1
            if zero_count > 0:
                print(f"  [WARNING] {title}: {num_layers_total} layers, {zero_count} zero weights detected")
        return
    
    # Verbose output - show detailed statistics
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


def print_lora_storage_info(save_path: str, title: str = "LoRA Storage Info", verbose: bool = False):
    """
    Print information about LoRA storage location and files.
    
    Args:
        save_path: Path to the LoRA adapter directory
        title: Title for the output
        verbose: If True, show detailed file information; if False, only show if path doesn't exist
    """
    if not os.path.exists(save_path):
        print(f"  [WARNING] {title}: Path does not exist - {save_path}")
        return
    
    if not verbose:
        # Silent mode - only check if path exists
        return
    
    # Verbose output - show detailed information
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    print(f"  Path: {save_path}")
    
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
        
        # Read and display adapter_config.json if it exists
        config_path = os.path.join(save_path, "adapter_config.json")
        if os.path.exists(config_path):
            import json
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"\n  adapter_config.json contents:")
                print(f"    r (rank): {config.get('r', 'N/A')}")
                print(f"    lora_alpha: {config.get('lora_alpha', 'N/A')}")
                print(f"    target_modules: {config.get('target_modules', 'N/A')}")
                print(f"    lora_dropout: {config.get('lora_dropout', 'N/A')}")
                print(f"    inference_mode: {config.get('inference_mode', 'N/A')}")
                print(f"    init_lora_weights: {config.get('init_lora_weights', 'N/A')}")
            except Exception as e:
                print(f"  [ERROR] Failed to read adapter_config.json: {e}")
    except OSError as e:
        print(f"  [ERROR] Cannot list directory: {e}")


def compare_lora_weights(weights1: Dict[str, Dict[str, torch.Tensor]], 
                         weights2: Dict[str, Dict[str, torch.Tensor]],
                         title: str = "LoRA Weight Comparison",
                         num_layers: int = 3,
                         verbose: bool = False):
    """
    Compare two sets of LoRA weights to check if they differ.
    
    Args:
        weights1: First set of LoRA weights
        weights2: Second set of LoRA weights
        title: Title for the output
        num_layers: Number of layers to display details for (only used if verbose=True)
        verbose: If True, show detailed comparison; if False, only show summary
    """
    layers = sorted(set(weights1.keys()) | set(weights2.keys()))
    total_diff = 0
    identical_count = 0
    total_compared = 0
    
    for layer_name in layers:
        w1 = weights1.get(layer_name, {})
        w2 = weights2.get(layer_name, {})
        
        for weight_type in ['lora_A', 'lora_B']:
            if weight_type in w1 and weight_type in w2:
                total_compared += 1
                diff = (w1[weight_type] - w2[weight_type]).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                total_diff += mean_diff
                
                if max_diff < 1e-8:
                    identical_count += 1
    
    if not verbose:
        # Only show summary
        if identical_count == total_compared:
            print(f"  [INFO] {title}: All weights identical ({identical_count}/{total_compared})")
        else:
            print(f"  [INFO] {title}: {total_compared - identical_count} weights differ, {identical_count} identical")
        return
    
    # Verbose output - show detailed comparison
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    
    displayed = 0
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
                
                if max_diff < 1e-8:
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
    num_layers: int = 3,
    verbose: bool = False
):
    """
    Compare base model weights with LoRA-merged model weights.
    Only compares the modules that LoRA targets.
    
    Args:
        base_model_state_dict: Base model weights
        merged_model_state_dict: LoRA-merged model weights
        target_modules: List of module names to compare
        num_layers: Number of layers to display details for (only used if verbose=True)
        verbose: If True, show detailed comparison; if False, only show summary
    """
    relevant_keys = []
    for key in base_model_state_dict.keys():
        if any(target in key for target in target_modules):
            # Skip lora-specific parameters
            if 'lora' not in key.lower():
                relevant_keys.append(key)
    
    total_diff = 0
    identical_count = 0
    
    for key in sorted(relevant_keys):
        if key not in merged_model_state_dict:
            continue
        
        base_w = base_model_state_dict[key]
        merged_w = merged_model_state_dict[key]
        
        diff = (base_w - merged_w).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        total_diff += mean_diff
        
        if max_diff < 1e-8:
            identical_count += 1
    
    lora_applied = len(relevant_keys) - identical_count
    
    if not verbose:
        # Only show summary
        if identical_count == len(relevant_keys):
            print(f"  [WARNING] LoRA merge check: All {len(relevant_keys)} layers identical - LoRA may not be applied!")
        else:
            print(f"  [INFO] LoRA merge check: {lora_applied}/{len(relevant_keys)} layers modified by LoRA")
        return
    
    # Verbose output - show detailed comparison
    print("\n" + "=" * 80)
    print(" Base Model vs LoRA-Merged Model Weight Comparison")
    print("=" * 80)
    
    print(f"\nComparing {len(relevant_keys)} relevant weight tensors (target modules: {target_modules})")
    
    displayed = 0
    
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
        
        if max_diff < 1e-8:
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
    print(f"  Different (LoRA applied): {lora_applied}")
    print(f"  Total mean difference across all layers: {total_diff:.8f}")
    
    if identical_count == len(relevant_keys):
        print("\n  [WARNING] All weights are identical - LoRA may not be properly applied!")


def print_adapter_info(model, title: str = "Adapter Information", verbose: bool = False):
    """
    Print information about loaded adapters in a PeftModel.
    
    Args:
        model: PeftModel instance
        title: Title for the output
        verbose: If True, show detailed adapter info; if False, show summary only
    """
    if not verbose:
        # Only show basic summary
        if hasattr(model, 'active_adapter'):
            print(f"  [INFO] Active adapter: {model.active_adapter}")
        return
    
    # Verbose output - show detailed information
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
                                  title: str = "Merged LoRA Weights Preview",
                                  verbose: bool = False):
    """
    Show what the merged LoRA weights would look like after add_weighted_adapter.
    
    Args:
        model: PeftModel instance
        adapter_names: List of adapter names to merge
        weights: List of weights for each adapter
        title: Title for the output
        verbose: If True, show detailed preview; if False, show summary only
    """
    if not verbose:
        return  # Silent in non-verbose mode
    
    # Verbose output
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


def debug_encode_before_training(model, init_adapter_path: str, verbose: bool = False):
    """
    Debug function to call before training in encode.py
    
    Args:
        model: PeftModel instance
        init_adapter_path: Path to initial adapter
        verbose: If True, show detailed output; if False, show summary only
    """
    if not verbose:
        return  # Silent in non-verbose mode
    
    print("\n" + "#" * 80)
    print(" DEBUG: ENCODE - BEFORE TRAINING")
    print("#" * 80)
    
    print_lora_storage_info(init_adapter_path, "Initial LoRA (base_weight) Storage Info", verbose=True)
    
    # Get initial LoRA weights
    lora_weights = get_lora_weights(model, "default")
    print_lora_weight_summary(lora_weights, "Initial LoRA Weights (from base_weight)", num_layers=3, verbose=True)
    
    return lora_weights


def debug_encode_after_training(model, save_path: str, initial_weights: Dict[str, Dict[str, torch.Tensor]], verbose: bool = False):
    """
    Debug function to call after training in encode.py
    
    Args:
        model: PeftModel instance
        save_path: Path where adapter was saved
        initial_weights: Initial LoRA weights before training
        verbose: If True, show detailed output; if False, show summary only
    """
    if not verbose:
        return  # Silent in non-verbose mode
    
    print("\n" + "#" * 80)
    print(" DEBUG: ENCODE - AFTER TRAINING")
    print("#" * 80)
    
    # Get trained LoRA weights
    trained_weights = get_lora_weights(model, "default")
    print_lora_weight_summary(trained_weights, "Trained LoRA Weights", num_layers=3, verbose=True)
    
    # Compare before and after
    compare_lora_weights(initial_weights, trained_weights, 
                        "Comparing Initial vs Trained LoRA Weights", num_layers=3, verbose=True)
    
    print_lora_storage_info(save_path, "Trained LoRA Storage Info (Save Location)", verbose=True)


def debug_inference_load_adapter(model, adapter_path: str, adapter_name: str, pid: int, verbose: bool = False):
    """
    Debug function to call after loading an adapter in inference.py
    
    Args:
        model: PeftModel instance
        adapter_path: Path to the adapter
        adapter_name: Name of the adapter
        pid: Passage ID
        verbose: If True, show detailed output; if False, show summary only
    """
    if not verbose:
        # Silent mode - only check for critical issues
        if not os.path.exists(adapter_path):
            print(f"  [ERROR] Adapter path does not exist: {adapter_path}")
            return
        
        # Quick check for zero weights
        lora_weights = get_lora_weights(model, adapter_name)
        if lora_weights:
            zero_count = sum(1 for weights in lora_weights.values() 
                           for wt in ['lora_B'] 
                           if wt in weights and weights[wt].abs().max().item() < ZERO_THRESHOLD)
            if zero_count > 0:
                print(f"  [WARNING] Adapter '{adapter_name}': {zero_count} zero lora_B weights detected")
        return
    
    # Verbose output
    print("\n" + "#" * 80)
    print(f" DEBUG: INFERENCE - LOADED ADAPTER (pid={pid})")
    print("#" * 80)
    
    print_lora_storage_info(adapter_path, f"Loaded LoRA Storage Info (Adapter '{adapter_name}')", verbose=True)
    
    # Directly read the safetensors file to see what's actually stored
    print("\n  [Direct file read - what's actually saved on disk:]")
    read_safetensors_file(adapter_path, f"Safetensors File for Adapter '{adapter_name}'", verbose=True)
    
    print_adapter_info(model, f"Adapter Info after loading '{adapter_name}'", verbose=True)
    
    # Get LoRA weights for this adapter (what's loaded in memory)
    print("\n  [Model memory - what's loaded into the model:]")
    lora_weights = get_lora_weights(model, adapter_name)
    print_lora_weight_summary(lora_weights, f"LoRA Weights for Adapter '{adapter_name}' (in memory)", num_layers=2, verbose=True)
    
    # Compare file vs memory to identify if loading is the problem
    compare_file_vs_memory(adapter_path, model, adapter_name, verbose=True)


def debug_inference_after_merge(model, adapter_names: List[str], 
                                 base_model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
                                 verbose: bool = False):
    """
    Debug function to call after merging adapters in inference.py
    
    Args:
        model: PeftModel instance
        adapter_names: List of adapter names that were merged
        base_model_state_dict: Optional base model weights for comparison
        verbose: If True, show detailed output; if False, show summary only
    """
    if not verbose:
        # Only show critical summary
        active = model.active_adapter if hasattr(model, 'active_adapter') else 'N/A'
        print(f"  [INFO] LoRA merge completed: {len(adapter_names)} adapters merged into '{active}'")
        
        # Quick check if LoRA was applied
        if base_model_state_dict is not None:
            current_state_dict = {k: v.clone() for k, v in model.state_dict().items() 
                                if 'lora' not in k.lower()}
            compare_model_weights_before_after_lora(
                base_model_state_dict, 
                current_state_dict,
                target_modules=['down_proj', 'gate_proj', 'up_proj'],
                num_layers=3,
                verbose=False
            )
        return
    
    # Verbose output
    print("\n" + "#" * 80)
    print(" DEBUG: INFERENCE - AFTER MERGING ADAPTERS")
    print("#" * 80)
    
    print(f"  Merged adapters: {adapter_names}")
    print(f"  Active adapter after merge: {model.active_adapter if hasattr(model, 'active_adapter') else 'N/A'}")
    
    # Get merged LoRA weights
    merged_weights = get_lora_weights(model, "merge")
    print_lora_weight_summary(merged_weights, "Merged LoRA Weights (adapter='merge')", num_layers=3, verbose=True)
    
    # Compare with base model if provided
    if base_model_state_dict is not None:
        # Get current model state dict (after merge)
        current_state_dict = {k: v.clone() for k, v in model.state_dict().items() 
                              if 'lora' not in k.lower()}
        compare_model_weights_before_after_lora(
            base_model_state_dict, 
            current_state_dict,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            num_layers=3,
            verbose=True
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


def read_safetensors_file(adapter_path: str, title: str = "Safetensors File Contents", verbose: bool = False):
    """
    Directly read and display the contents of a safetensors file to verify what's actually saved.
    This is useful for debugging when model loading seems to not work correctly.
    
    Args:
        adapter_path: Path to the adapter directory
        title: Title for the output
        verbose: If True, show detailed file contents; if False, only check for issues
    """
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    
    if not os.path.exists(safetensors_path):
        if verbose or not os.path.exists(adapter_path):
            print(f"  [ERROR] File not found: {safetensors_path}")
        return None
    
    if not SAFETENSORS_AVAILABLE:
        if verbose:
            print("  [ERROR] safetensors library not installed. Run: pip install safetensors")
        return None
    
    try:
        weights = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = tensor
            
            if not verbose:
                # Quick check for zero lora_B weights
                zero_lora_b = sum(1 for key in f.keys() 
                                if 'lora_B' in key and f.get_tensor(key).abs().max().item() < ZERO_THRESHOLD)
                if zero_lora_b > 0:
                    print(f"  [WARNING] {title}: {zero_lora_b} zero lora_B weights in file")
                return weights
            
            # Verbose output
            print("\n" + "=" * 80)
            print(f" {title}")
            print("=" * 80)
            print(f"  File: {safetensors_path}")
            print(f"  Keys in file: {list(f.keys())}")
            
            for key in f.keys():
                tensor = weights[key]
                
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
        if verbose:
            print(f"  [ERROR] Failed to read safetensors file: {e}")
        return None


def verify_saved_adapter(save_path: str, verbose: bool = False):
    """
    Verify that the adapter was saved correctly by reading the safetensors file directly.
    This should be called after model.save_pretrained() to confirm the weights are saved.
    
    Args:
        save_path: Path to the saved adapter
        verbose: If True, show detailed verification; if False, show summary only
    """
    weights = read_safetensors_file(save_path, f"Saved Adapter Contents at {save_path}", verbose=verbose)
    
    if not weights:
        return
    
    # Summarize lora_B status
    lora_b_keys = [k for k in weights.keys() if 'lora_B' in k]
    lora_b_zeros = sum(1 for k in lora_b_keys if weights[k].abs().max().item() < ZERO_THRESHOLD)
    
    if not verbose:
        if lora_b_zeros == len(lora_b_keys):
            print(f"  [ERROR] All {len(lora_b_keys)} lora_B tensors are zeros - training may not have saved correctly!")
        elif lora_b_zeros > 0:
            print(f"  [WARNING] {lora_b_zeros}/{len(lora_b_keys)} lora_B tensors are zeros")
        return
    
    # Verbose output
    print("\n" + "#" * 80)
    print(" VERIFICATION: Checking saved adapter file")
    print("#" * 80)
    
    print(f"\n  [SUMMARY]")
    print(f"    Total lora_B tensors: {len(lora_b_keys)}")
    print(f"    All-zero lora_B tensors: {lora_b_zeros}")
    
    if lora_b_zeros == len(lora_b_keys):
        print(f"    [PROBLEM] All lora_B tensors are zeros - training may not have saved correctly!")
    elif lora_b_zeros > 0:
        print(f"    [WARNING] Some lora_B tensors are zeros")
    else:
        print(f"    [OK] All lora_B tensors have trained values")


def compare_file_vs_memory(adapter_path: str, model, adapter_name: str, verbose: bool = False):
    """
    Compare LoRA weights between the safetensors file and what's loaded in model memory.
    This helps diagnose if the issue is in saving or loading.
    
    Args:
        adapter_path: Path to the adapter
        model: PeftModel instance
        adapter_name: Name of the adapter
        verbose: If True, show detailed comparison; if False, show summary only
    """
    # Read from file
    file_weights = read_safetensors_file(adapter_path, "File Contents (disk)", verbose=False)
    if file_weights is None:
        if verbose:
            print("  [ERROR] Could not read file weights")
        return
    
    # Get from memory
    memory_weights = get_lora_weights(model, adapter_name)
    
    # Compare lora_B values specifically
    mismatches = 0
    for file_key in file_weights.keys():
        if 'lora_B' not in file_key:
            continue
            
        file_tensor = file_weights[file_key]
        file_is_zero = file_tensor.abs().max().item() < ZERO_THRESHOLD
        
        # Find corresponding memory tensor
        found_in_memory = False
        for mem_layer, mem_weights in memory_weights.items():
            if 'lora_B' in mem_weights:
                mem_tensor = mem_weights['lora_B']
                # Check if shapes match
                if file_tensor.shape == mem_tensor.shape:
                    mem_is_zero = mem_tensor.abs().max().item() < ZERO_THRESHOLD
                    
                    if file_is_zero != mem_is_zero:
                        mismatches += 1
                        if verbose:
                            print(f"\n    [MISMATCH] {file_key}")
                            print(f"      File: {'ZERO' if file_is_zero else 'NON-ZERO'} (max={file_tensor.abs().max().item():.8f})")
                            print(f"      Memory: {'ZERO' if mem_is_zero else 'NON-ZERO'} (max={mem_tensor.abs().max().item():.8f})")
                    
                    found_in_memory = True
                    break
        
        if not found_in_memory and verbose:
            print(f"\n    [NOT FOUND IN MEMORY] {file_key}")
    
    if not verbose:
        if mismatches > 0:
            print(f"  [ERROR] File vs memory mismatch: {mismatches} lora_B tensors differ - PEFT may not be loading weights correctly!")
        return
    
    # Verbose output
    print("\n" + "=" * 80)
    print(f" COMPARISON: File vs Memory for adapter '{adapter_name}'")
    print("=" * 80)
    
    print(f"\n  File has {len(file_weights)} tensors")
    print(f"  Memory has {len(memory_weights)} layer groups")
    print(f"\n  [lora_B Comparison]")
    
    if mismatches == 0:
        print(f"\n    [OK] File and memory weights match in zero/non-zero status")
    else:
        print(f"\n    [PROBLEM] {mismatches} weight tensors have different zero/non-zero status between file and memory!")
        print(f"    This suggests PEFT is not loading the weights correctly.")
