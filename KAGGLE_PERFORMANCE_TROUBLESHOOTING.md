# Kaggle性能差异排查完全指南

## 问题描述

用户按照文档在Kaggle上运行了encode和inference，但得到的结果与论文相差很多。本文档分析可能的原因并提供解决方案。

---

## 🔍 常见原因分析

### 1. **样本数量过少（最可能原因）**

您的代码中没有看到完整的参数，但如果使用了`sample=3`或其他小值：

**问题**：
- LoRA权重只训练了3个文档的知识
- Inference时遇到的大部分问题无对应的LoRA权重
- 论文使用**全量数据**（2WikiMultihopQA约2500条）

**解决方案**：
```python
args_encode = argparse.Namespace(
    # ...
    sample=None,  # ✅ 使用全量数据！不要限制sample
    # ...
)
```

**验证方法**：
```python
# 检查生成了多少个LoRA权重
!find /kaggle/working/PRAG/offline -name "adapter_model.safetensors" | wc -l
# 应该是：数据量 × 3（每个数据3个passage的LoRA）
```

---

### 2. **data_type参数错误**

**问题**：可能使用了`data_type=None`而不是`data_type='total'`

**影响**：
- `data_type=None`：分别处理各类型文件（inference/compositional等），结果会分散
- `data_type='total'`：统一处理所有数据，与论文一致

**解决方案**：
```python
# Encode和Inference都使用
data_type='total'  # ✅ 正确
# data_type=None   # ❌ 错误
```

---

### 3. **测试集与训练集不匹配**

**检查点**：
- ✅ Encode和Inference使用**相同的JSON文件路径**
- ✅ Encode和Inference使用**相同的data_type**
- ✅ Inference的`sample`参数 ≤ Encode的`sample`（如果两者都设置了）

**错误示例**：
```python
# ❌ 训练用compositional，测试用total
encode: data_type='compositional', sample=100
inference: data_type='total', sample=500  # 测试集包含训练集没见过的类型
```

**正确示例**：
```python
# ✅ 训练和测试一致
encode: data_type='total', sample=None
inference: data_type='total', sample=None
```

---

### 4. **训练不充分**

**问题**：可能只训练了1个epoch或很少的steps

**论文设置**（根据代码默认值）：
```python
num_train_epochs=1        # 1个epoch通常足够（文档很少）
learning_rate=0.0003      # 标准学习率
per_device_train_batch_size=1  # batch size通常为1
```

**如果数据量很大，可能需要**：
- 增加epoch数（2-3）
- 或确保每个文档都被充分训练

**验证训练日志**：
```
### Solving total ###  ← 应该看到这个
100%|██████████| N/N [时间<时间, 速度it/s]  ← N应该是数据总量
```

---

### 5. **inference_method选择错误**

**三种方法对比**：

| Method | 使用LoRA | 使用文档 | 预期性能 |
|--------|----------|----------|---------|
| `icl` | ❌ | ✅ | 基线（论文中较低）|
| **`prag`** | ✅ | ❌ | **论文主方法**（应该最好）|
| `combine` | ✅ | ✅ | 组合方法 |

**常见错误**：
```python
inference_method='icl'  # ❌ 这是基线，不使用LoRA！
inference_method='prag'  # ✅ 论文主方法
```

---

### 6. **参数不一致**

**必须一致的参数**（Encode ↔ Inference）：

| 参数 | 说明 | 检查 |
|------|------|------|
| `model_name` | 模型名称 | ✅ 必须完全相同 |
| `dataset` | 数据集 | ✅ 必须相同 |
| `data_type` | 数据类型 | ✅ 必须相同 |
| `augment_model` | 增强模型 | ✅ 必须相同 |
| `with_cot` | 是否CoT | ✅ 必须相同 |
| `lora_rank` | LoRA秩 | ✅ 必须相同 |
| `lora_alpha` | LoRA alpha | ✅ 必须相同 |
| `learning_rate` | 学习率 | ✅ 必须相同 |
| `num_train_epochs` | Epoch数 | ✅ 必须相同 |

**检查方法**：
```python
# 在Inference前添加验证
print("=== 参数检查 ===")
print(f"Encode路径: {encode的LoRA路径}")
print(f"Inference路径: {inference要加载的路径}")
# 两者应该完全匹配！
```

---

### 7. **模型版本不匹配**

**问题**：使用了与增强数据不同的模型

**检查**：
```python
# 增强数据文件夹名称
data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/

# 与运行时的model_name必须一致
model_name='qwen2.5-1.5b-instruct'  # ✅ 匹配
model_name='llama3.2-1b-instruct'   # ❌ 不匹配
```

---

### 8. **max_new_tokens设置过小**

**问题**：生成的答案被截断

**检查**：
```python
max_new_tokens=50  # ❌ 可能太小（论文可能用更大值）
max_new_tokens=100  # ✅ 论文推荐值（根据数据集调整）
```

**2WikiMultihopQA建议**：`max_new_tokens=100`（答案通常较短）

---

## 📋 完整诊断清单

### 第一步：检查Encode参数

```python
# 在您的代码中添加：
print("=== Encode参数检查 ===")
print(f"dataset: {args_encode.dataset}")
print(f"data_type: {args_encode.data_type}")  # 应该是 'total'
print(f"sample: {args_encode.sample}")  # 应该是 None 或 很大的数
print(f"model_name: {args_encode.model_name}")
print(f"augment_model: {args_encode.augment_model}")
print(f"with_cot: {args_encode.with_cot}")
print(f"lora_rank: {args_encode.lora_rank}")
print(f"lora_alpha: {args_encode.lora_alpha}")
print(f"num_train_epochs: {args_encode.num_train_epochs}")
print(f"learning_rate: {args_encode.learning_rate}")
```

### 第二步：检查生成的LoRA数量

```python
import glob
lora_paths = glob.glob(f'{WORK_DIR}/offline/**/adapter_model.safetensors', recursive=True)
print(f"\n=== 生成的LoRA权重数量: {len(lora_paths)} ===")

# 期望值：如果sample=None，应该有 数据总量 × 3 个
# 例如：2WikiMultihopQA全量约2500条 → 应该有 ~7500 个LoRA文件
```

### 第三步：检查Inference参数

```python
print("\n=== Inference参数检查 ===")
print(f"inference_method: {args_inference.inference_method}")  # 应该是 'prag'
print(f"sample: {args_inference.sample}")  # 应该 ≤ encode的sample
print(f"max_new_tokens: {args_inference.max_new_tokens}")

# 所有其他参数应与Encode一致
print("\n=== 与Encode对比 ===")
params_to_check = ['dataset', 'data_type', 'model_name', 'augment_model', 
                   'with_cot', 'lora_rank', 'lora_alpha', 
                   'learning_rate', 'num_train_epochs']
for param in params_to_check:
    encode_val = getattr(args_encode, param)
    inference_val = getattr(args_inference, param)
    match = "✅" if encode_val == inference_val else "❌"
    print(f"{match} {param}: encode={encode_val}, inference={inference_val}")
```

### 第四步：检查结果文件

```python
# 查看结果
result_path = glob.glob(f'{WORK_DIR}/output/**/result.txt', recursive=True)
if result_path:
    print("\n=== Inference结果 ===")
    with open(result_path[0], 'r') as f:
        print(f.read())
else:
    print("❌ 未找到result.txt文件！")
```

---

## 🎯 推荐的完整配置

### Encode配置（全量数据）

```python
import argparse

args_encode = argparse.Namespace(
    # 基础配置
    model_name='qwen2.5-1.5b-instruct',  # 与数据文件夹名称一致
    dataset='2wikimultihopqa',
    data_type='total',  # ✅ 使用total
    
    # 数据配置
    sample=None,  # ✅ 全量数据！
    augment_model='qwen2.5-1.5b-instruct',  # 与model_name相同
    
    # 训练配置
    with_cot=True,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    
    # LoRA配置
    lora_rank=2,
    lora_alpha=32,
)
```

### Inference配置（与Encode一致）

```python
args_inference = argparse.Namespace(
    # 必须与Encode完全一致的参数
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',
    augment_model='qwen2.5-1.5b-instruct',
    with_cot=True,
    lora_rank=2,
    lora_alpha=32,
    learning_rate=0.0003,
    num_train_epochs=1,
    
    # Inference特有参数
    inference_method='prag',  # ✅ 论文主方法
    sample=None,  # 全量测试
    max_new_tokens=100,  # 根据数据集调整
)
```

---

## 🔧 快速修复脚本

将以下代码添加到您的notebook开头：

```python
# === 诊断脚本 ===
def diagnose_setup():
    """诊断Encode/Inference配置"""
    import glob
    
    print("=" * 50)
    print("PRAG配置诊断")
    print("=" * 50)
    
    # 1. 检查数据文件
    data_path = f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}'
    json_files = glob.glob(f'{data_path}/*.json')
    print(f"\n✓ 数据文件 ({len(json_files)}个):")
    for f in json_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"  - {os.path.basename(f)}: {size_kb:.1f}KB")
    
    # 2. 检查LoRA权重
    lora_files = glob.glob(f'{WORK_DIR}/offline/**/adapter_model.safetensors', recursive=True)
    print(f"\n✓ LoRA权重: {len(lora_files)}个")
    if len(lora_files) > 0:
        print(f"  示例路径: {lora_files[0]}")
    
    # 3. 预估期望性能
    if len(lora_files) < 1000:
        print(f"\n⚠️  警告: LoRA数量较少({len(lora_files)})，可能是sample设置过小")
        print("   建议: 使用 sample=None 训练全量数据")
    
    # 4. 检查输出目录
    output_files = glob.glob(f'{WORK_DIR}/output/**/result.txt', recursive=True)
    if output_files:
        print(f"\n✓ 找到结果文件: {output_files[0]}")
    else:
        print("\n⚠️  未找到result.txt，inference可能未运行或失败")
    
    print("=" * 50)

# 在Encode前运行
diagnose_setup()
```

---

## 📊 性能预期

### 论文报告的性能（2WikiMultihopQA）

根据PRAG论文（具体数值需查看论文Table）：

- **ICL（基线）**：EM约20-30%
- **PRAG（论文方法）**：EM约40-50%（显著提升）
- **Combine**：EM可能介于两者之间或更高

### 如果您的结果

**情况1：EM < 10%**
- 🔴 **严重问题**：可能sample过小、参数不匹配、或inference_method错误

**情况2：EM 10-20%**
- 🟡 **需改进**：可能训练数据不足或配置有误

**情况3：EM > 30%**
- 🟢 **合理**：接近论文性能，可能是数据集版本或模型差异

---

## 🚀 重新运行建议

如果确认有问题，建议：

### 1. 清理旧数据

```python
# 清理旧的LoRA和输出
!rm -rf {WORK_DIR}/offline
!rm -rf {WORK_DIR}/output
print("✓ 已清理旧数据")
```

### 2. 使用完整配置重新运行

```python
# === Cell: Encode（全量数据）===
args_encode = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',  # ← 关键
    sample=None,  # ← 关键：全量数据
    augment_model='qwen2.5-1.5b-instruct',
    with_cot=True,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
)

# 手动设置augment_model（避免None问题）
if args_encode.augment_model is None:
    args_encode.augment_model = args_encode.model_name

print("开始Encode（预计耗时：数小时，取决于数据量）...")
encode_main(args_encode)
print("✓ Encode完成！")

# === Cell: 诊断 ===
diagnose_setup()

# === Cell: Inference ===
args_inference = argparse.Namespace(
    # 与Encode完全一致
    model_name=args_encode.model_name,
    dataset=args_encode.dataset,
    data_type=args_encode.data_type,
    augment_model=args_encode.augment_model,
    with_cot=args_encode.with_cot,
    lora_rank=args_encode.lora_rank,
    lora_alpha=args_encode.lora_alpha,
    learning_rate=args_encode.learning_rate,
    num_train_epochs=args_encode.num_train_epochs,
    
    # Inference特有
    inference_method='prag',  # ← 关键
    sample=None,  # 全量测试
    max_new_tokens=100,
)

print("开始Inference...")
inference_main(args_inference)
print("✓ Inference完成！")

# === Cell: 查看结果 ===
result_path = glob.glob(f'{WORK_DIR}/output/**/result.txt', recursive=True)
if result_path:
    print("\n=== 最终结果 ===")
    with open(result_path[0], 'r') as f:
        print(f.read())
```

---

## 📞 进一步排查

如果按照上述步骤仍有问题，请提供：

1. **完整的Encode日志**（特别是`### Solving xxx ###`部分）
2. **LoRA数量统计**
3. **完整的Inference日志**
4. **result.txt内容**
5. **实际使用的完整参数配置**

这将帮助精确定位问题。

---

## 总结：最可能的3个原因

根据经验，99%的性能差异来自：

1. ✅ **sample参数设置过小**（如sample=3而非None）
2. ✅ **data_type使用错误**（None而非'total'）
3. ✅ **inference_method错误**（icl而非prag）

**立即检查这三点！**
