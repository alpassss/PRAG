#!/usr/bin/env python3
"""
检查LoRA权重是否真的不同

这个脚本会：
1. 加载base_weight和几个passage的adapter权重
2. 比较它们是否相同
3. 检查训练是否实际改变了权重
"""

import os
import sys
import torch
from safetensors.torch import load_file

def compare_adapters(base_path, sample_id=0, num_passages=3):
    """比较base_weight和训练后的adapters"""
    
    print("="*80)
    print("检查LoRA权重是否发生变化")
    print("="*80)
    
    # Load base weight
    base_weight_path = os.path.join(base_path, "base_weight", "adapter_model.safetensors")
    
    if not os.path.exists(base_weight_path):
        print(f"❌ Base weight不存在: {base_weight_path}")
        return False
    
    print(f"\n加载Base Weight: {base_weight_path}")
    base_weights = load_file(base_weight_path)
    print(f"  包含 {len(base_weights)} 个参数")
    
    # Sample a few keys
    sample_keys = list(base_weights.keys())[:5]
    print(f"  示例参数名: {sample_keys}")
    
    # Calculate base weight statistics
    base_stats = {}
    for key, tensor in base_weights.items():
        base_stats[key] = {
            'mean': tensor.float().mean().item(),
            'std': tensor.float().std().item(),
            'abs_sum': tensor.float().abs().sum().item()
        }
    
    # Check first few adapters
    print(f"\n检查Sample {sample_id}的Adapters:")
    
    all_identical = True
    differences_found = []
    
    for pid in range(num_passages):
        adapter_path = os.path.join(base_path, "total", f"data_{sample_id}", f"passage_{pid}", "adapter_model.safetensors")
        
        if not os.path.exists(adapter_path):
            print(f"\n  ❌ Passage {pid}: 文件不存在 - {adapter_path}")
            continue
        
        print(f"\n  Passage {pid}: {adapter_path}")
        
        # Load adapter weights
        adapter_weights = load_file(adapter_path)
        print(f"    包含 {len(adapter_weights)} 个参数")
        
        # Compare with base weight
        identical_params = 0
        different_params = 0
        max_diff = 0.0
        
        for key in base_weights.keys():
            if key not in adapter_weights:
                print(f"    ⚠️ 参数 {key} 在adapter中不存在")
                continue
            
            base_tensor = base_weights[key]
            adapter_tensor = adapter_weights[key]
            
            if base_tensor.shape != adapter_tensor.shape:
                print(f"    ⚠️ 参数 {key} 形状不匹配: {base_tensor.shape} vs {adapter_tensor.shape}")
                continue
            
            # Check if identical
            diff = (base_tensor - adapter_tensor).abs().max().item()
            max_diff = max(max_diff, diff)
            
            if diff < 1e-8:  # Essentially identical
                identical_params += 1
            else:
                different_params += 1
        
        total_params = identical_params + different_params
        print(f"    参数统计:")
        print(f"      总参数: {total_params}")
        print(f"      相同参数: {identical_params} ({100*identical_params/total_params:.1f}%)")
        print(f"      不同参数: {different_params} ({100*different_params/total_params:.1f}%)")
        print(f"      最大差异: {max_diff:.6e}")
        
        if different_params == 0:
            print(f"    ❌ 所有参数与base_weight完全相同！训练没有效果！")
        elif different_params < total_params * 0.1:
            print(f"    ⚠️ 只有很少参数改变，训练可能有问题")
            all_identical = False
        else:
            print(f"    ✓ 参数已被训练改变")
            all_identical = False
            differences_found.append(pid)
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    if all_identical:
        print("❌ 所有adapters与base_weight完全相同！")
        print("\n可能的原因：")
        print("  1. 训练数据为空或格式错误")
        print("  2. 训练循环没有执行（num_train_epochs=0或数据为空）")
        print("  3. 梯度没有反向传播")
        print("  4. 优化器没有更新参数")
        print("\n建议：")
        print("  1. 检查encode输出，确认训练循环执行了")
        print("  2. 在encode.py的训练循环中添加loss打印")
        print("  3. 确认augment数据中有valid的QA对")
        return False
    elif differences_found:
        print(f"✓ 找到 {len(differences_found)} 个训练过的adapters: {differences_found}")
        print("\n但是如果COMBINE结果仍等于ICL，可能原因：")
        print("  1. 训练的知识不足以改善结果")
        print("  2. 训练数据质量不好")
        print("  3. 需要更多训练轮数（增加num_train_epochs）")
        print("  4. 学习率太小或太大")
        return True
    else:
        print("⚠️ 检测到问题但不完全相同")
        return False


def check_adapter_content(adapter_path):
    """详细检查单个adapter的内容"""
    print("\n" + "="*80)
    print(f"详细检查Adapter: {adapter_path}")
    print("="*80)
    
    if not os.path.exists(adapter_path):
        print(f"❌ 文件不存在")
        return
    
    weights = load_file(adapter_path)
    
    print(f"\n参数列表 (共 {len(weights)} 个):")
    for i, (key, tensor) in enumerate(weights.items()):
        if i < 10 or i >= len(weights) - 5:  # Show first 10 and last 5
            mean = tensor.float().mean().item()
            std = tensor.float().std().item()
            abs_sum = tensor.float().abs().sum().item()
            print(f"  {key}")
            print(f"    Shape: {tuple(tensor.shape)}, Dtype: {tensor.dtype}")
            print(f"    Mean: {mean:.6f}, Std: {std:.6f}, |Sum|: {abs_sum:.6f}")
        elif i == 10:
            print(f"  ... (省略 {len(weights) - 15} 个参数)")
    
    # Check if all weights are zero
    all_zero = all(tensor.abs().sum().item() < 1e-8 for tensor in weights.values())
    if all_zero:
        print("\n❌ 所有权重都是零！适配器未被训练！")
    else:
        print("\n✓ 权重包含非零值")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检查LoRA权重是否真的训练了")
    parser.add_argument("--base_path", type=str, required=True,
                       help="Base path, e.g., offline/qwen2.5-1.5b-instruct/rank=2_alpha=32/complexwebquestions/lr=0.0003_epoch=1_direct/aug_model=qwen2.5-1.5b-instruct")
    parser.add_argument("--sample_id", type=int, default=0,
                       help="Which sample to check (default: 0)")
    parser.add_argument("--num_passages", type=int, default=3,
                       help="Number of passages to check (default: 3)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed content of first adapter")
    
    args = parser.parse_args()
    
    # Check if adapters are different from base
    result = compare_adapters(args.base_path, args.sample_id, args.num_passages)
    
    # If requested, show detailed content
    if args.detailed:
        adapter_path = os.path.join(args.base_path, "total", f"data_{args.sample_id}", "passage_0", "adapter_model.safetensors")
        check_adapter_content(adapter_path)
    
    sys.exit(0 if result else 1)
