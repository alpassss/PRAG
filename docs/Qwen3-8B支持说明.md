# Qwen3-8B 模型支持修复说明

## 问题描述

用户尝试测试 Qwen3-8B 模型时遇到以下问题：

1. 错误地将 `qwen2.5-1.5b-instruct` 映射到 `Qwen/Qwen3-8B`
2. 使用了已弃用的 `torch_dtype` 参数（应使用 `dtype`）
3. 缺少独立的 qwen3-8b 模型名称映射

## 修复内容

### 1. 恢复 qwen2.5-1.5b-instruct 原始映射

**修复前**:
```python
elif model_name == "qwen2.5-1.5b-instruct":
    return "Qwen/Qwen3-8B"  # 错误！这会导致混淆
```

**修复后**:
```python
elif model_name == "qwen2.5-1.5b-instruct":
    return "Qwen/Qwen2.5-1.5B-Instruct"  # 恢复原始映射
```

### 2. 添加新的 qwen3-8b 映射

```python
elif model_name == "qwen3-8b":
    return "Qwen/Qwen3-8B"  # 新增独立的 Qwen3-8B 映射
```

### 3. 修复 torch_dtype 弃用警告

**修复前**:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32,  # 已弃用
    ...
)
```

**修复后**:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=torch.float32,  # 使用新参数名
    ...
)
```

## 使用方法

### 使用 Qwen3-8B 模型

现在可以使用新的模型名称 `qwen3-8b` 来测试 Qwen3-8B：

```bash
python src/encode.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --sample 10 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot
```

### 使用 Qwen2.5-1.5B 模型（原有功能保持不变）

```bash
python src/encode.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset 2wikimultihopqa \
    --sample 10 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32
```

## 支持的模型列表

| 模型名称 | HuggingFace 路径 |
|---------|------------------|
| `llama3-8b-instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `qwen2.5-1.5b-instruct` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen3-8b` | `Qwen/Qwen3-8B` |
| `llama3.2-1b-instruct` | `meta-llama/Llama-3.2-1B-Instruct` |
| 其他 | 直接使用输入的名称作为路径 |

## 注意事项

1. **模型名称要匹配**: 使用 `qwen3-8b` 而不是 `qwen2.5-1.5b-instruct` 来测试 Qwen3-8B
2. **数据增强模型**: 如果不指定 `--augment_model`，默认会使用与 `--model_name` 相同的模型
3. **版本兼容性**: 修复后的代码兼容最新版本的 transformers 和 torch

## 技术细节

### dtype vs torch_dtype

在 transformers 4.30+ 版本中，`torch_dtype` 参数已被弃用，应使用 `dtype` 代替：

- **旧版本**: `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)`
- **新版本**: `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float32)`

这个变化确保了与最新版本的 transformers 库的兼容性。

## 错误排查

如果仍然遇到问题，请检查：

1. **模型名称是否正确**: 使用 `qwen3-8b` 而不是 `qwen2.5-1.5b-instruct`
2. **transformers 版本**: 确保使用的是兼容的版本
3. **数据路径**: 确保 `data_aug/{dataset}/{model_name}/` 目录下有对应的数据文件

---

**文档版本**: 1.0  
**创建日期**: 2026-01-28  
**修复提交**: 待提交
