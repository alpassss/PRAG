# Qwen3-8B 模型支持问题修复

## 问题分析

您遇到的问题有三个根本原因：

### 1. 模型名称映射错误 ❌
您将 `qwen2.5-1.5b-instruct` 映射到了 `Qwen/Qwen3-8B`，这导致：
- 模型名称与实际模型不匹配
- 可能导致数据路径错误（因为数据文件夹是按模型名称组织的）
- 代码混乱，难以维护

### 2. torch_dtype 已弃用 ⚠️
错误信息显示：`torch_dtype` is deprecated! Use `dtype` instead!
- 在 transformers 4.30+ 版本中，参数名从 `torch_dtype` 改为 `dtype`
- 虽然暂时还能工作，但会产生警告，未来版本可能不再支持

### 3. 缺少 Qwen3-8B 的独立映射 ❌
原代码没有为 Qwen3-8B 提供独立的模型名称

## 修复方案

### ✅ 修复 1: 恢复原始映射并添加新条目

**src/utils.py** 中的 `get_model_path` 函数：

```python
def get_model_path(model_name):
    if model_name == "llama3-8b-instruct": 
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5-1.5b-instruct":
        return "Qwen/Qwen2.5-1.5B-Instruct"  # ✅ 恢复原始映射
    elif model_name == "qwen3-8b":           # ✅ 新增 Qwen3-8B
        return "Qwen/Qwen3-8B"
    elif model_name == "llama3.2-1b-instruct":
        return "meta-llama/Llama-3.2-1B-Instruct"
    else:
        return model_name
```

### ✅ 修复 2: 更新为新的参数名

**src/utils.py** 中的 `get_model` 函数：

```python
def get_model(model_name, max_new_tokens=20):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.float32,  # ✅ 改为 dtype
        low_cpu_mem_usage=True,
        device_map="auto", 
        trust_remote_code=True
    )
    ...
```

## 如何使用

### 测试 Qwen3-8B 模型

现在使用 `qwen3-8b` 作为模型名称：

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

### 继续使用 Qwen2.5-1.5B 模型

原有功能保持不变：

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

## 重要说明

### 数据文件路径

数据增强文件应该位于：
- `data_aug/2wikimultihopqa/qwen3-8b/` （用于 Qwen3-8B）
- `data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/` （用于 Qwen2.5-1.5B）

如果您已经用 `qwen2.5-1.5b-instruct` 名称生成了 Qwen3-8B 的数据，您需要：

**选项 A**: 重命名数据文件夹
```bash
mv data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct \
   data_aug/2wikimultihopqa/qwen3-8b
```

**选项 B**: 重新运行数据增强
```bash
python src/augment.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300 \
    --topk 3
```

## 验证修复

1. **检查模型路径映射**：
```python
from utils import get_model_path

print(get_model_path("qwen3-8b"))  
# 输出: Qwen/Qwen3-8B

print(get_model_path("qwen2.5-1.5b-instruct"))  
# 输出: Qwen/Qwen2.5-1.5B-Instruct
```

2. **检查无警告**：
运行时不应再看到 "torch_dtype is deprecated" 警告

3. **检查模型加载**：
模型应该能正确下载并加载

## 支持的所有模型

| 模型名称 | HuggingFace 路径 | 说明 |
|---------|------------------|------|
| `llama3-8b-instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` | Llama 3 8B |
| `qwen2.5-1.5b-instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | Qwen 2.5 1.5B |
| `qwen3-8b` | `Qwen/Qwen3-8B` | Qwen 3 8B ✨ 新增 |
| `llama3.2-1b-instruct` | `meta-llama/Llama-3.2-1B-Instruct` | Llama 3.2 1B |

## 后续步骤

1. 使用 `--model_name qwen3-8b` 运行您的实验
2. 如果遇到数据路径问题，检查 `data_aug/` 目录结构
3. 确保您的 transformers 和 torch 版本是最新的并兼容

## 需要帮助？

如果还有问题，请检查：
- 模型名称拼写是否正确
- 数据文件是否在正确的路径
- 网络连接是否正常（下载 Qwen3-8B 模型需要良好的网络）
- HuggingFace 是否可以正常访问

---

**修复提交**: 15abd5d  
**文档**: docs/Qwen3-8B支持说明.md
