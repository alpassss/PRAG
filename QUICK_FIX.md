# 🚨 快速修复指南

## 您的错误

```
### Solving total ###
  0%|                                                    | 0/10 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/kaggle/working/PRAG/src/encode.py", line 241, in <module>
```

## ⚡ 快速修复（3步）

### 第 1 步: 检查数据路径

```bash
ls data_aug/2wikimultihopqa/
```

您会看到什么？

**情况 A**: 看到 `qwen2.5-1.5b-instruct/` 文件夹
→ 跳到方案 A

**情况 B**: 看到 `qwen3-8b/` 文件夹  
→ 跳到方案 B

**情况 C**: 什么都没有或错误
→ 跳到方案 C

---

### 方案 A: 您有 qwen2.5-1.5b-instruct 数据

#### 选项 1: 重命名为 qwen3-8b（如果想用 Qwen3-8B）

```bash
mv data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct \
   data_aug/2wikimultihopqa/qwen3-8b

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

#### 选项 2: 继续使用原模型名称

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

---

### 方案 B: 您有 qwen3-8b 数据

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

---

### 方案 C: 没有数据或路径错误

#### 步骤 1: 先生成数据

```bash
python src/augment.py \
    --model_name qwen3-8b \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300 \
    --topk 3
```

#### 步骤 2: 然后训练

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

---

## 🎯 记住这个规则

```
模型名称参数 = 数据文件夹名称
```

| 命令中使用的 --model_name | 必须存在的数据路径 |
|-------------------------|------------------|
| `qwen3-8b` | `data_aug/2wikimultihopqa/qwen3-8b/` |
| `qwen2.5-1.5b-instruct` | `data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/` |

**不匹配 = 错误！**

---

## ✅ 验证修复

运行这个测试命令（只处理 1 个数据）：

```bash
python src/encode.py \
    --model_name YOUR_MODEL_NAME \
    --dataset 2wikimultihopqa \
    --data_type total \
    --sample 1 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32
```

**成功**: 进度条会到 100%
```
### Solving total ###
100%|████████████████████| 1/1 [XX:XX<00:00, XX.XXs/it]
```

**失败**: 还是报错 → 查看详细文档

---

## 📚 需要更多帮助？

- **简单解答**: `问题解答.md`
- **详细诊断**: `ERROR_DIAGNOSIS.md`
- **完整修复**: `FIX_SUMMARY_CN.md`

---

**最后更新**: 2026-01-28  
**提交**: 90a1fab
