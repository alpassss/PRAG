"""
模型对比验证工具 - 确认LoRA权重是否在inference时生效

这个脚本比较基础模型和加载LoRA适配器后的模型，验证参数是否发生了改变。
如果LoRA正常工作，参数应该会有差异。
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os


def load_base_model(model_name):
    """
    加载基础模型（不含LoRA）
    
    参数:
        model_name: 模型名称，如 'qwen2.5-1.5b-instruct'
    
    返回:
        model: 基础模型
    """
    print(f"\n{'='*60}")
    print("步骤1: 加载基础模型（不含LoRA）")
    print(f"{'='*60}")
    
    model_path = f"Qwen/{model_name.replace('.', '-').replace('qwen', 'Qwen')}"
    print(f"模型路径: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✓ 基础模型加载成功")
    return model


def load_lora_model(model_name, adapter_path):
    """
    加载带LoRA适配器的模型
    
    参数:
        model_name: 模型名称
        adapter_path: LoRA适配器路径
    
    返回:
        model: 加载了LoRA的模型
    """
    print(f"\n{'='*60}")
    print("步骤2: 加载带LoRA适配器的模型")
    print(f"{'='*60}")
    
    # 先加载基础模型
    model_path = f"Qwen/{model_name.replace('.', '-').replace('qwen', 'Qwen')}"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"LoRA适配器路径: {adapter_path}")
    
    # 检查路径是否存在
    if not os.path.exists(adapter_path):
        print(f"❌ 错误: 路径不存在!")
        return None
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        model, 
        adapter_path,
        is_trainable=False
    )
    
    print("✓ LoRA适配器加载成功")
    return model


def compare_models(base_model, lora_model, sample_layers=None):
    """
    比较两个模型的参数
    
    参数:
        base_model: 基础模型
        lora_model: 加载LoRA的模型
        sample_layers: 采样层数（None表示检查所有层）
    
    返回:
        dict: 比较结果统计
    """
    print(f"\n{'='*60}")
    print("步骤3: 比较模型参数")
    print(f"{'='*60}")
    
    base_params = dict(base_model.named_parameters())
    lora_params = dict(lora_model.named_parameters())
    
    total_params = 0
    changed_params = 0
    max_diff = 0.0
    layer_count = 0
    
    print("\n正在比较参数...")
    
    for name, base_param in base_params.items():
        # 如果指定了采样层数，只检查前N层
        if sample_layers is not None:
            layer_num = None
            if 'layers.' in name:
                try:
                    layer_num = int(name.split('layers.')[1].split('.')[0])
                except:
                    pass
            
            if layer_num is not None and layer_num >= sample_layers:
                continue
        
        total_params += 1
        
        # 在LoRA模型中找对应参数
        if name in lora_params:
            lora_param = lora_params[name]
            
            # 比较参数值
            if base_param.shape == lora_param.shape:
                diff = (base_param - lora_param).abs().max().item()
                max_diff = max(max_diff, diff)
                
                # 如果有差异（大于极小阈值），说明参数改变了
                if diff > 1e-8:
                    changed_params += 1
                    
                    # 显示前几个改变的参数
                    if changed_params <= 5:
                        print(f"  参数 {name[:50]}... 发生改变")
                        print(f"    最大差异: {diff:.6f}")
        
        layer_count += 1
        
        # 每100个参数显示一次进度
        if layer_count % 100 == 0:
            print(f"  已检查 {layer_count} 个参数...")
    
    return {
        'total_params': total_params,
        'changed_params': changed_params,
        'max_diff': max_diff
    }


def print_results(stats):
    """
    打印比较结果
    
    参数:
        stats: 统计结果字典
    """
    print(f"\n{'='*60}")
    print("步骤4: 验证结果")
    print(f"{'='*60}")
    
    print(f"\n总参数数量: {stats['total_params']}")
    print(f"改变的参数: {stats['changed_params']}")
    print(f"最大差异值: {stats['max_diff']:.6f}")
    
    print(f"\n{'='*60}")
    if stats['changed_params'] > 0:
        print("✓ ✓ ✓ LoRA适配器已激活 - 模型参数发生了改变")
        print(f"{'='*60}")
        print("\n✅ 结论: LoRA权重在inference时正在生效")
        print(f"   {stats['changed_params']} 个参数已被LoRA修改")
        print(f"   这证明加载的不是基础模型，而是LoRA修改后的模型")
    else:
        print("❌ ❌ ❌ LoRA适配器未激活 - 所有参数都相同")
        print(f"{'='*60}")
        print("\n❌ 结论: LoRA权重可能没有正确加载")
        print("   可能的原因:")
        print("   1. adapter_path路径错误")
        print("   2. 适配器文件损坏")
        print("   3. 加载代码有问题")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='验证LoRA权重是否在inference时生效'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='模型名称，如 qwen2.5-1.5b-instruct'
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        required=True,
        help='LoRA适配器路径，如 offline/.../data_0/passage_0'
    )
    parser.add_argument(
        '--sample_layers',
        type=int,
        default=None,
        help='只检查前N层（可选，用于快速测试）'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("模型对比验证工具")
    print("="*60)
    print("\n目的: 验证LoRA适配器是否在inference时正确加载并修改模型参数")
    print("\n工作原理:")
    print("  1. 加载基础模型（不含LoRA）")
    print("  2. 加载相同模型但带LoRA适配器")
    print("  3. 比较两个模型的参数")
    print("  4. 如果参数有差异 → LoRA正在工作 ✓")
    print("     如果参数完全相同 → LoRA未加载 ❌")
    
    try:
        # 加载基础模型
        base_model = load_base_model(args.model_name)
        
        # 加载LoRA模型
        lora_model = load_lora_model(args.model_name, args.adapter_path)
        
        if lora_model is None:
            print("\n❌ 无法加载LoRA模型，请检查adapter_path是否正确")
            return
        
        # 比较模型
        stats = compare_models(base_model, lora_model, args.sample_layers)
        
        # 打印结果
        print_results(stats)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
