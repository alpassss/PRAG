# 错误诊断和解决方案

## 🔍 问题分析

您遇到的错误发生在 `src/encode.py` 的第 241 行，当程序尝试遍历数据时：

```
### Solving total ###
  0%|                                                    | 0/10 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/kaggle/working/PRAG/src/encode.py", line 241, in <module>
    main(args)
```

### 可能的原因

根据代码分析，错误最可能发生在以下几个地方：

#### 1. **数据路径问题** (最可能 ⭐⭐⭐⭐⭐)

**问题位置**: `src/utils.py` 第 80 行的 `load_data` 函数

```python
def load_data(data_name, data_type, model_name):
    solve_dataset = []
    input_dir = os.path.join(DATA_ROOT_DIR, data_name, model_name)
    files = [f for f in os.listdir(input_dir)]  # ← 这里可能出错
```

**原因**: 
- 如果您使用了新的模型名称 `qwen3-8b`，但数据增强文件还在旧路径
- 期望路径: `data_aug/2wikimultihopqa/qwen3-8b/`
- 实际路径可能: `data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/`

**错误信息**: 
```python
FileNotFoundError: [Errno 2] No such file or directory: 'data_aug/2wikimultihopqa/qwen3-8b'
```

#### 2. **数据格式问题** (可能 ⭐⭐⭐)

**问题位置**: `src/encode.py` 第 207 行

```python
augment = data["augment"]  # ← 可能 KeyError
```

**原因**: 数据文件中可能缺少 `augment` 字段

**错误信息**:
```python
KeyError: 'augment'
```

#### 3. **模型名称不匹配** (可能 ⭐⭐)

如果您使用 `--model_name qwen2.5-1.5b-instruct` 但期望加载 Qwen3-8B，会导致路径混乱。

## ✅ 解决方案

### 方案 1: 检查并修复数据路径 (推荐)

#### 步骤 1: 检查当前数据路径

```bash
# 查看您的数据增强文件在哪里
ls -la data_aug/2wikimultihopqa/

# 示例输出可能是：
# drwxr-xr-x 2 user user 4096 Jan 28 qwen2.5-1.5b-instruct/
# drwxr-xr-x 2 user user 4096 Jan 28 llama3-8b-instruct/
```

#### 步骤 2: 确认您要使用的模型

如果您想使用 **Qwen3-8B**：

```bash
# 选项 A: 重命名现有数据文件夹
mv data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct \
   data_aug/2wikimultihopqa/qwen3-8b

# 选项 B: 重新生成数据（如果数据是用其他模型生成的）
python src/augment.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300 \
    --topk 3
```

#### 步骤 3: 使用正确的模型名称运行

```bash
python src/encode.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --data_type total \
    --sample 10 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot
```

### 方案 2: 使用现有的 Qwen2.5-1.5B 数据

如果您的数据是用 `qwen2.5-1.5b-instruct` 生成的，就继续使用这个模型名称：

```bash
python src/encode.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset 2wikimultihopqa \
    --data_type total \
    --sample 10 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32
```

### 方案 3: 检查数据文件完整性

如果以上都不行，检查数据文件格式：

```bash
# 查看数据文件内容
cat data_aug/2wikimultihopqa/YOUR_MODEL_NAME/total.json | head -50

# 检查是否有 "augment" 字段
python3 << 'EOF'
import json
with open('data_aug/2wikimultihopqa/YOUR_MODEL_NAME/total.json', 'r') as f:
    data = json.load(f)
    print(f"总数据量: {len(data)}")
    if len(data) > 0:
        print(f"第一条数据的键: {data[0].keys()}")
        if 'augment' in data[0]:
            print(f"augment 字段存在，包含 {len(data[0]['augment'])} 个段落")
        else:
            print("❌ 错误: 缺少 'augment' 字段")
EOF
```

## 🔧 调试步骤

### 1. 完整错误信息

如果错误仍然存在，运行以下命令获取完整错误信息：

```bash
python src/encode.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --data_type total \
    --sample 10 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot 2>&1 | tee error_log.txt
```

### 2. 检查数据文件结构

```bash
# 查看完整的数据目录结构
tree data_aug/2wikimultihopqa/ -L 2

# 或者使用 find
find data_aug/2wikimultihopqa/ -type f -name "*.json" | head -10
```

### 3. 验证模型路径映射

```python
# 测试模型路径映射是否正确
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from utils import get_model_path

print("测试模型路径映射:")
print(f"qwen3-8b -> {get_model_path('qwen3-8b')}")
print(f"qwen2.5-1.5b-instruct -> {get_model_path('qwen2.5-1.5b-instruct')}")
EOF
```

## 📊 常见错误模式

### 错误 1: FileNotFoundError

```
FileNotFoundError: [Errno 2] No such file or directory: 
'data_aug/2wikimultihopqa/qwen3-8b'
```

**原因**: 数据文件夹不存在  
**解决**: 使用方案 1 的步骤 2

### 错误 2: KeyError

```
KeyError: 'augment'
```

**原因**: 数据文件格式不正确  
**解决**: 重新生成数据增强文件

### 错误 3: 空数据列表

```
### Solving total ###
  0%|                                                    | 0/0 [00:00<?, ?it/s]
```

**原因**: 数据文件为空或格式错误  
**解决**: 检查数据文件内容

## 🎯 快速修复清单

- [ ] 检查数据路径是否存在: `ls data_aug/2wikimultihopqa/`
- [ ] 确认使用的模型名称与数据路径匹配
- [ ] 如果使用 `qwen3-8b`，确保数据在 `data_aug/.../qwen3-8b/`
- [ ] 如果使用 `qwen2.5-1.5b-instruct`，确保数据在 `data_aug/.../qwen2.5-1.5b-instruct/`
- [ ] 检查数据文件是否包含 `augment` 字段
- [ ] 运行完整命令获取详细错误信息

## 📝 模型名称与路径对应表

| 命令行参数 (`--model_name`) | 数据路径 | HuggingFace 路径 |
|----------------------------|---------|------------------|
| `qwen2.5-1.5b-instruct` | `data_aug/{dataset}/qwen2.5-1.5b-instruct/` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen3-8b` | `data_aug/{dataset}/qwen3-8b/` | `Qwen/Qwen3-8B` |
| `llama3-8b-instruct` | `data_aug/{dataset}/llama3-8b-instruct/` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `llama3.2-1b-instruct` | `data_aug/{dataset}/llama3.2-1b-instruct/` | `meta-llama/Llama-3.2-1B-Instruct` |

## 💡 最佳实践

1. **始终保持模型名称一致**: 数据增强、训练、推理都使用相同的模型名称
2. **先生成数据**: 在运行 `encode.py` 之前，确保已经运行过 `augment.py`
3. **使用小样本测试**: 先用 `--sample 1` 测试，确保流程正确

---

**问题已解决？**
- ✅ 是：太好了！您现在可以继续训练
- ❌ 否：请提供完整的错误信息，我们会继续帮助您

**相关文档**:
- `FIX_SUMMARY_CN.md` - Qwen3-8B 修复总结
- `docs/Qwen3-8B支持说明.md` - 详细使用说明
