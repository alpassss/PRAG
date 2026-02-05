# 🎯 Qwen3-8B 支持修复 - 完整总结

## 📌 您的问题

您尝试测试 Qwen3-8B 模型时遇到以下问题：

```python
# 您的修改（有问题）
def get_model_path(model_name):
    if model_name == "llama3-8b-instruct": 
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5-1.5b-instruct":
        return "Qwen/Qwen3-8B"  # ❌ 问题：名称不匹配
```

然后运行命令：
```bash
python src/encode.py --model_name qwen2.5-1.5b-instruct ...
```

**出现的错误**：
1. ⚠️ 警告：`torch_dtype` is deprecated! Use `dtype` instead!
2. ❌ 模型名称与实际模型不匹配
3. ❌ 可能导致数据路径问题

## ✅ 解决方案

### 修复 1: 正确的模型映射

```python
def get_model_path(model_name):
    if model_name == "llama3-8b-instruct": 
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "qwen2.5-1.5b-instruct":
        return "Qwen/Qwen2.5-1.5B-Instruct"  # ✅ 恢复原始
    elif model_name == "qwen3-8b":            # ✅ 新增专用名称
        return "Qwen/Qwen3-8B"
    elif model_name == "llama3.2-1b-instruct":
        return "meta-llama/Llama-3.2-1B-Instruct"
    else:
        return model_name
```

### 修复 2: 更新弃用的参数

```python
# 修改前
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32,  # ❌ 已弃用
    ...
)

# 修改后
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=torch.float32,  # ✅ 新参数
    ...
)
```

## 🚀 现在如何使用

### 方案 A: 使用 Qwen3-8B（推荐）

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

### 方案 B: 使用 Qwen2.5-1.5B（原有功能）

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

## ⚠️ 重要：数据文件路径

### 如果您已经生成了数据

如果您已经用 `qwen2.5-1.5b-instruct` 名称为 Qwen3-8B 生成了数据增强文件，需要重命名：

```bash
# 检查当前数据路径
ls data_aug/2wikimultihopqa/

# 如果存在 qwen2.5-1.5b-instruct 文件夹但里面是 Qwen3-8B 的数据
# 方案 A: 重命名文件夹
mv data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct \
   data_aug/2wikimultihopqa/qwen3-8b

# 方案 B: 重新生成数据
python src/augment.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300 \
    --topk 3
```

### 数据路径对应关系

| 模型名称 | 数据路径 |
|---------|---------|
| qwen2.5-1.5b-instruct | `data_aug/{dataset}/qwen2.5-1.5b-instruct/` |
| qwen3-8b | `data_aug/{dataset}/qwen3-8b/` |
| llama3-8b-instruct | `data_aug/{dataset}/llama3-8b-instruct/` |
| llama3.2-1b-instruct | `data_aug/{dataset}/llama3.2-1b-instruct/` |

## ✨ 修复效果

### 之前
```
Namespace(model_name='qwen2.5-1.5b-instruct', ...)
`torch_dtype` is deprecated! Use `dtype` instead!  ⚠️
Warning: You are sending unauthenticated requests...
model.safetensors.index.json: 32.9kB [00:00, 90.7MB/s]
Downloading: 16.4G/16.4G [00:08<00:00, 2.01GB/s]  # 下载 Qwen3-8B
...
[可能出现路径错误或其他问题]
```

### 之后
```
Namespace(model_name='qwen3-8b', ...)
# 没有 torch_dtype 警告！ ✅
Warning: You are sending unauthenticated requests...
model.safetensors.index.json: 32.9kB [00:00, 90.7MB/s]
Downloading: 16.4G/16.4G [00:08<00:00, 2.01GB/s]  # 下载 Qwen3-8B
Loading weights: 100%|█| 399/399 [00:04<00:00, 97.18it/s]
No LoRA base weight, creating...
Save LoRA base weight to .../qwen3-8b/rank=2_alpha=32/base_weight
### Solving total ###
[正常运行] ✅
```

## 📚 所有支持的模型

| 模型名称 | HuggingFace 路径 | 模型大小 | 状态 |
|---------|------------------|---------|------|
| llama3-8b-instruct | meta-llama/Meta-Llama-3-8B-Instruct | ~8B | ✅ |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | ~1.5B | ✅ |
| qwen3-8b | Qwen/Qwen3-8B | ~8B | ✨ 新增 |
| llama3.2-1b-instruct | meta-llama/Llama-3.2-1B-Instruct | ~1B | ✅ |

## 🔍 验证修复

### 1. 检查语法
```bash
cd /home/runner/work/PRAG/PRAG
python -m py_compile src/utils.py
# 应该没有错误
```

### 2. 测试模型路径映射
```python
from src.utils import get_model_path

# 测试各个模型
print(get_model_path("qwen3-8b"))
# 输出: Qwen/Qwen3-8B ✅

print(get_model_path("qwen2.5-1.5b-instruct"))
# 输出: Qwen/Qwen2.5-1.5B-Instruct ✅
```

### 3. 运行测试
```bash
# 小规模测试
python src/encode.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --sample 1 \
    --num_train_epochs 1 \
    --lora_rank 2 \
    --lora_alpha 32
```

## 📄 相关文档

- 📖 详细文档: `docs/Qwen3-8B支持说明.md`
- 🔧 完整解决方案: `SOLUTION.md`
- 📝 变更摘要: `CHANGES_SUMMARY.md`

## 💡 提示

1. **首次运行**: Qwen3-8B 是 8B 参数模型，下载需要约 16GB 空间和良好的网络
2. **HuggingFace Token**: 设置 `HF_TOKEN` 环境变量可以提高下载速度
3. **GPU 内存**: 确保有足够的 GPU 内存（推荐 16GB+）
4. **数据准备**: 确保先运行数据增强生成 QA 对

## 🆘 故障排查

### 问题：仍然看到 torch_dtype 警告
**解决**：确保您拉取了最新的代码（提交 15abd5d）

### 问题：找不到数据文件
**解决**：检查 `data_aug/{dataset}/qwen3-8b/` 路径是否存在

### 问题：下载速度慢
**解决**：
```bash
# 设置 HuggingFace token
export HF_TOKEN=your_token_here

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题：GPU 内存不足
**解决**：使用较小的模型或减少 batch size

## 📊 完整工作流程

```bash
# 1. 数据增强（如果还没有）
python src/augment.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300 \
    --topk 3

# 2. 训练 LoRA 参数
python src/encode.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --sample 300 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot

# 3. 推理测试
python src/inference.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --max_new_tokens 128 \
    --sample 100 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --inference_method prag
```

---

## 📌 快速参考

**修复提交**: `15abd5d`  
**修复分支**: `copilot/simplify-lora-debug-output`  
**修复文件**: `src/utils.py`  
**修复日期**: 2026-01-28  

**核心变更**:
- ✅ 添加 `qwen3-8b` 模型映射
- ✅ 恢复 `qwen2.5-1.5b-instruct` 原始映射
- ✅ 修复 `torch_dtype` → `dtype`
- ✅ 添加完整中文文档

**使用建议**:
- 🎯 使用 `--model_name qwen3-8b` 测试 Qwen3-8B
- 📁 确保数据路径正确对应
- 🚀 享受无警告的运行体验！

---

如有任何问题，请查看详细文档或提出 issue。祝您使用愉快！🎉
