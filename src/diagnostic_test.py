#!/usr/bin/env python3
"""
Diagnostic script to help debug PRAG inference issues.
This script checks:
1. Whether LoRA weight files exist
2. Whether data is loaded correctly
3. Whether adapters are loaded and activated properly
"""

import os
import sys
import json
import torch
import argparse
from peft import PeftModel

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from root_dir_path import ROOT_DIR
from utils import load_data, get_model


def check_lora_weights_exist(args):
    """Check if LoRA weight files exist for the given configuration."""
    print("\n" + "="*80)
    print("CHECKING LORA WEIGHT FILES")
    print("="*80)
    
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
    
    print(f"Base adapter path: {load_adapter_path}")
    
    if not os.path.exists(load_adapter_path):
        print(f"❌ ERROR: Base adapter path does not exist!")
        return False
    
    # Load data to check for first few samples
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    filename, fulldata = data_list[0]
    filename = filename.split('.')[0]
    
    print(f"\nChecking weights for dataset: {filename}")
    print(f"Number of test samples: {len(fulldata)}")
    
    # Check first few samples
    check_samples = min(3, len(fulldata))
    all_exist = True
    
    for test_id in range(check_samples):
        data = fulldata[test_id]
        passages = data["passages"]
        num_passages = len(passages)
        
        print(f"\n  Sample {test_id}:")
        print(f"    Question: {data['question'][:60]}...")
        print(f"    Number of passages: {num_passages}")
        
        for pid in range(num_passages):
            adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
            adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
            
            if os.path.exists(adapter_file):
                file_size = os.path.getsize(adapter_file)
                print(f"    ✓ Passage {pid}: {adapter_file}")
                print(f"      File size: {file_size:,} bytes")
            else:
                print(f"    ❌ Passage {pid}: NOT FOUND - {adapter_file}")
                all_exist = False
    
    if all_exist:
        print(f"\n✅ All checked LoRA weight files exist!")
    else:
        print(f"\n❌ Some LoRA weight files are missing!")
    
    return all_exist


def check_data_loading(args):
    """Check if data is loaded correctly."""
    print("\n" + "="*80)
    print("CHECKING DATA LOADING")
    print("="*80)
    
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    
    print(f"Number of data files: {len(data_list)}")
    
    for filename, fulldata in data_list:
        print(f"\nFile: {filename}")
        print(f"  Number of samples: {len(fulldata)}")
        
        if len(fulldata) > 0:
            sample = fulldata[0]
            print(f"  First sample keys: {list(sample.keys())}")
            
            if 'augment' in sample:
                print(f"  ✓ Has 'augment' field")
                print(f"  Number of augments: {len(sample['augment'])}")
                
                if 'passages' in sample:
                    print(f"  Number of passages: {len(sample['passages'])}")
                    if len(sample['passages']) == len(sample['augment']):
                        print(f"  ✓ Passages and augments count match")
                    else:
                        print(f"  ❌ Mismatch: {len(sample['passages'])} passages vs {len(sample['augment'])} augments")
            else:
                print(f"  ❌ Missing 'augment' field!")
                return False
    
    print(f"\n✅ Data loading appears correct!")
    return True


def check_adapter_loading(args):
    """Check if adapters can be loaded and activated."""
    print("\n" + "="*80)
    print("CHECKING ADAPTER LOADING")
    print("="*80)
    
    # Load model
    print("Loading base model...")
    model, tokenizer, generation_config = get_model(args.model_name)
    print(f"✓ Model loaded: {type(model).__name__}")
    
    # Load data
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    filename, fulldata = data_list[0]
    filename = filename.split('.')[0]
    
    # Get adapter path
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
    
    # Try loading adapters for first sample
    test_id = 0
    data = fulldata[test_id]
    passages = data["passages"]
    
    print(f"\nTrying to load adapters for sample {test_id}...")
    print(f"  Number of passages: {len(passages)}")
    
    try:
        for pid in range(len(passages)):
            adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
            
            if pid == 0:
                print(f"\n  Loading first adapter (pid={pid})...")
                model = PeftModel.from_pretrained(
                    model, 
                    adapter_path,
                    is_trainable = False
                )
                # Get the default adapter name that was assigned
                first_adapter_name = model.active_adapter if hasattr(model, 'active_adapter') else "default"
                print(f"    ✓ Loaded with adapter name: '{first_adapter_name}'")
                print(f"    Model type: {type(model).__name__}")
            else:
                print(f"\n  Loading additional adapter (pid={pid})...")
                model.load_adapter(adapter_path, adapter_name = f"adapter_{pid}")
                print(f"    ✓ Loaded as adapter 'adapter_{pid}'")
        
        # Get list of all loaded adapter names
        if hasattr(model, 'peft_config'):
            adapter_names = list(model.peft_config.keys())
            print(f"\n  All loaded adapters: {adapter_names}")
        else:
            adapter_names = ["default"] + [f"adapter_{i}" for i in range(1, len(passages))]
            print(f"\n  Assuming adapter names: {adapter_names}")
        
        # Check active adapters
        if hasattr(model, 'active_adapters'):
            print(f"  Active adapters: {model.active_adapters}")
        
        # Try merging adapters (using same method as inference.py)
        print(f"\n  Merging adapters...")
        num_adapters = len(passages)
        model.add_weighted_adapter(
            adapters = adapter_names[:num_adapters], 
            weights = [1.0 / num_adapters] * num_adapters,
            adapter_name = "merge", 
            combination_type = "linear",  # Using linear to match the fix in inference.py
        )
        print(f"    ✓ Created 'merge' adapter (linear combination with normalized weights)")
        
        model.set_adapter("merge")
        print(f"    ✓ Set 'merge' as active adapter")
        
        if hasattr(model, 'active_adapters'):
            print(f"  Active adapters after merge: {model.active_adapters}")
        
        # Check if model parameters have changed
        print(f"\n  Checking model parameters...")
        peft_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable parameters: {peft_params:,}")
        
        print(f"\n✅ Adapter loading and merging successful!")
        
        # Clean up
        model.delete_adapter("merge")
        model = model.unload()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading adapters: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Diagnostic script for PRAG inference issues")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--augment_model", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    
    args = parser.parse_args()
    
    if args.augment_model is None:
        args.augment_model = args.model_name
    
    print("="*80)
    print("PRAG DIAGNOSTIC TEST")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"\nROOT_DIR: {ROOT_DIR}")
    
    # Run checks
    checks = {
        "Data Loading": check_data_loading(args),
        "LoRA Weights Exist": check_lora_weights_exist(args),
        "Adapter Loading": check_adapter_loading(args),
    }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name}: {status}")
    
    if all(checks.values()):
        print(f"\n✅ All checks passed! The setup appears correct.")
        print(f"\nIf you're still experiencing issues, the problem might be:")
        print(f"  1. The LoRA weights contain insufficient information (training issue)")
        print(f"  2. The model generation is not using the adapters correctly")
        print(f"  3. The prompts are not constructed correctly")
    else:
        print(f"\n❌ Some checks failed. Please fix the issues above.")
    
    print("="*80)


if __name__ == "__main__":
    main()
