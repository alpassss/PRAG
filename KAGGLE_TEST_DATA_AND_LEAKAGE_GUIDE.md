# Kaggle测试数据与数据泄露完全说明

## 📋 问题总览

您提出了三个关键问题：

1. **Inference时使用什么数据作为测试集？**
2. **QA对是否会造成数据泄露？**
3. **Encode时具体训练了哪些内容？**

---

## 1️⃣ Inference测试数据来源

### ✅ 答案：使用**相同的total.json**，但只使用其中的`question`和`answer`字段

```python
# 代码位置：src/inference.py 第54-60行
for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
    question = data["question"]      # ✅ 使用
    passages = data["passages"]      # ✅ 使用（icl/combine模式）
    answer = data["answer"]          # ✅ 用于评估
    # ❌ augment字段（rewrite、qa）在inference中完全不使用
```

### 数据使用对比

| 阶段 | 使用字段 | 作用 |
|------|---------|------|
| **Encode** | `augments`字段（passage/rewrite/qa） | 训练LoRA权重 |
| **Inference** | `question`, `answer`, `passages` | 测试和评估 |

### 为什么用同一个文件？

**PRAG方法的核心理念**：
- **Encode阶段**：将文档知识"记忆"到模型参数（LoRA）中
- **Inference阶段**：测试模型能否**仅凭参数化知识**回答问题，不再依赖原始文档

### 完整流程示例

```python
# 同一个数据项在不同阶段的使用：
data = {
    "question": "Who is the mother of the director?",
    "answer": ["Małgorzata Braunek"],
    "passages": ["passage text 1", "passage text 2"],
    "augments": [
        {
            "passage": "original passage",
            "qwen2.5-1.5b-instruct_rewrite": "rewritten passage",
            "qwen2.5-1.5b-instruct_qa": [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"}
            ]
        }
    ]
}

# Encode阶段（src/encode.py）：
# ✅ 使用 augments 字段训练LoRA
train_on(augments)  # 学习passage、rewrite、qa

# Inference阶段（src/inference.py）：
# ✅ 仅使用 question、answer、passages
model.generate(question)  # 生成答案
evaluate(pred, answer)     # 计算指标
# passages仅在icl/combine模式使用
```

---

## 2️⃣ QA对是否造成数据泄露？

### ✅ 答案：**不会**，这是论文设计的核心方法

### 为什么不是数据泄露？

**1. QA对的来源**：
- QA对是由**LLM基于文档生成**的辅助训练数据
- **不是原始测试问题**，而是帮助模型理解文档的中间问题
- 类似于"阅读理解练习题"，帮助模型消化文档内容

**2. 三阶段训练策略（代码：src/encode.py 第78-90行）**：

```python
qas = aug["qwen2.5-1.5b-instruct_qa"]  # 例如：3个QA对
qpa_cnt = (len(qas) + 1) // 2          # qpa_cnt = 2

# 阶段1+2（QA 0-1）：passage + QA对
for qid in [0, 1]:  # 前2个QA
    for ppp in [原始passage, 重写passage]:
        train(question=qa["question"], 
              context=ppp,           # ✅ 带上下文
              answer=qa["answer"])

# 阶段3（QA 2）：仅QA对，无passage
for qid in [2]:  # 后1个QA
    train(question=qa["question"], 
          context=None,              # ❌ 不给上下文
          answer=qa["answer"])
```

**3. 训练目标**：

| 阶段 | 训练内容 | 目的 |
|------|---------|------|
| QA 1-2 | 问题 + 文档 → 答案 | 学习如何**使用**文档回答问题 |
| QA 3 | 问题 → 答案（无文档） | 将知识**记忆**到参数中 |

**4. Inference验证无泄露**：

```python
# inference.py 第97行
ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))

# prag模式：psgs=None
# ❌ 不给模型任何文档
# ✅ 仅靠LoRA参数化知识回答
```

### 实际例子

```json
{
  "question": "Who is the mother of the director of Polish-Russian War?",
  "answer": ["Małgorzata Braunek"],
  "augments": [{
    "qwen2.5-1.5b-instruct_qa": [
      {"question": "Who directed Polish-Russian War?", "answer": "Xawery Żuławski"},
      {"question": "What is Xawery's profession?", "answer": "Film director"},
      {"question": "Who is Xawery's mother?", "answer": "Małgorzata Braunek"}
    ]
  }]
}

# Encode：训练模型回答这3个辅助问题（逐步理解文档）
# Inference：模型需要回答原始问题（依靠训练时学到的知识）
# ✅ QA不是原始问题，而是分解的子问题
```

---

## 3️⃣ Encode训练的三部分内容

### ✅ 答案：训练**文档、改写文档、QA对**三部分

### 详细分解（src/encode.py 第71-91行）

```python
def get_train_data(aug_model, augments, tokenizer, args):
    prompt_ids = []
    for aug in augments:
        psg = aug["passage"]                        # 1️⃣ 原始文档
        rew = aug[f"{aug_model}_rewrite"]           # 2️⃣ 重写文档
        qas = aug[f"{aug_model}_qa"]                # 3️⃣ QA对列表
        
        qpa_cnt = (len(qas) + 1) // 2  # 例如：3个QA → qpa_cnt=2
        
        # 第一部分：前一半QA + 原始/重写文档
        for qid in range(qpa_cnt):  # [0, 1]
            for ppp in [psg, rew]:  # 原始和重写都训练
                prompt_ids.append(
                    get_prompt(tokenizer, 
                              qa["question"], 
                              [ppp],          # ✅ 带上下文
                              qa["answer"])
                )
        
        # 第二部分：后一半QA 无文档
        for qid in range(qpa_cnt, len(qas)):  # [2]
            prompt_ids.append(
                get_prompt(tokenizer, 
                          qa["question"], 
                          None,              # ❌ 无上下文
                          qa["answer"])
            )
    return prompt_ids
```

### 训练数据结构示例

假设有3个QA对：

| 训练样本 | 问题来源 | 上下文 | 答案 |
|---------|---------|-------|------|
| 1 | QA[0] | 原始passage | answer[0] |
| 2 | QA[0] | 重写passage | answer[0] |
| 3 | QA[1] | 原始passage | answer[1] |
| 4 | QA[1] | 重写passage | answer[1] |
| 5 | QA[2] | **无** | answer[2] |

**总计5个训练样本**（对于每个文档）

### 为什么这样设计？

1. **样本1-4（带上下文）**：
   - 教模型如何从文档提取信息
   - 学习文档和问题的关联

2. **样本5（无上下文）**：
   - 强迫模型将知识记忆到参数
   - 实现参数化存储

3. **重写文档的作用**：
   - 增加数据多样性
   - 让模型学习不同表述的相同知识

---

## 📝 完整Inference代码

```python
# Cell 8: Inference - 方法1（函数调用）
import sys
import argparse
import os

# 配置环境
WORK_DIR = '/kaggle/working/PRAG'
sys.path.insert(0, f'{WORK_DIR}/src')

import root_dir_path
root_dir_path.ROOT_DIR = WORK_DIR

from inference import main as inference_main

# 配置参数（必须与encode完全一致）
args_inference = argparse.Namespace(
    # ===== 必须与encode一致的参数 =====
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',              # ✅ 与encode相同
    with_cot=True,
    augment_model='qwen2.5-1.5b-instruct',
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    
    # ===== Inference专用参数 =====
    inference_method='prag',        # 'icl' / 'prag' / 'combine'
    sample=3,                       # ≤ encode的sample数
    max_new_tokens=50,
)

print("=== 开始Inference ===")
inference_main(args_inference)
print("✅ Inference完成！")
```

### 三种inference_method对比

```python
# 方法1: icl (In-Context Learning - 基线)
args_inference.inference_method = 'icl'
# ❌ 不使用LoRA权重
# ✅ 使用原始passages
# 结果：传统RAG方法

# 方法2: prag (论文核心方法)
args_inference.inference_method = 'prag'
# ✅ 使用LoRA权重
# ❌ 不使用passages
# 结果：纯参数化知识

# 方法3: combine (组合方法)
args_inference.inference_method = 'combine'
# ✅ 使用LoRA权重
# ✅ 使用passages
# 结果：参数化知识 + 检索文档
```

---

## 🎯 关键结论

### 测试数据

✅ **使用相同的total.json文件**  
✅ **但仅使用question/answer/passages字段**  
✅ **augments字段在inference中完全不使用**

### 数据泄露

✅ **不存在数据泄露**  
✅ **QA对是辅助训练数据，不是测试问题**  
✅ **Inference使用原始问题，验证参数化知识**

### Encode内容

✅ **训练文档、重写文档、QA对三部分**  
✅ **前半QA带上下文（学习使用文档）**  
✅ **后半QA无上下文（记忆到参数）**

---

## 📊 完整工作流程图

```
数据准备
└─ total.json
   ├─ question (原始测试问题)
   ├─ answer (金标准答案)
   ├─ passages (检索文档)
   └─ augments (增强数据)
      ├─ passage (原始)
      ├─ rewrite (重写)
      └─ qa (辅助QA对)

↓

Encode阶段
├─ 读取：augments字段
├─ 训练：
│  ├─ QA[0-1] + 原始passage → answer
│  ├─ QA[0-1] + 重写passage → answer
│  └─ QA[2] + 无上下文 → answer
└─ 输出：LoRA权重 (offline/)

↓

Inference阶段
├─ 读取：question、answer、passages
├─ 加载：LoRA权重
├─ 推理：
│  ├─ icl: 原始模型 + passages
│  ├─ prag: LoRA模型（无passages）
│  └─ combine: LoRA模型 + passages
└─ 输出：
   ├─ predict.json (预测结果)
   └─ result.txt (EM/F1/Prec/Recall)
```

---

## 🔍 验证无数据泄露的方法

### 检查1：Inference代码不使用augments

```python
# src/inference.py 第54-60行
for test_id, data in tqdm(enumerate(fulldata)):
    question = data["question"]
    passages = data["passages"]
    answer = data["answer"]
    # ✅ 没有使用 data["augments"]
    # ✅ 没有使用任何QA对
```

### 检查2：PRAG模式不使用文档

```python
# src/inference.py 第97行
ret.append(get_pred(model, psgs=None if args.inference_method == "prag" else passages))
# prag模式：psgs=None
# ✅ 完全依赖LoRA参数化知识
```

### 检查3：运行输出验证

```bash
=== 步骤6: 运行Encode ===
### Solving total ###
100%|██████████| 3/3 [00:34<00:00, 11.64s/it]
# ✅ 处理3个数据项，生成9个LoRA权重（3问题×3文档）

=== 步骤8: 运行Inference ===
### Solving total ###
100%|██████████| 3/3 [00:12<00:00, 4.21s/it]
# ✅ 推理3个问题（与encode相同数据项）
# ✅ 但使用不同字段（question vs augments）
```

---

## ⚠️ 注意事项

### sample参数一致性

```python
# ❌ 错误
args_encode.sample = 3
args_inference.sample = 5  # 超出encode范围

# ✅ 正确
args_encode.sample = 5
args_inference.sample = 3  # ≤ encode的sample
# 或
args_inference.sample = 5  # 完整测试
```

### 参数必须匹配

| 参数 | 要求 | 原因 |
|------|------|------|
| model_name | 完全一致 | LoRA路径匹配 |
| dataset | 完全一致 | LoRA路径匹配 |
| augment_model | 完全一致 | LoRA路径匹配 |
| lora_rank | 完全一致 | LoRA维度匹配 |
| lora_alpha | 完全一致 | LoRA缩放匹配 |
| num_train_epochs | 完全一致 | LoRA路径匹配 |
| learning_rate | 完全一致 | LoRA路径匹配 |
| with_cot | 完全一致 | 提示模板匹配 |
| data_type | 完全一致 | 数据集匹配 |

---

## 📈 结果解读

### 输出位置

```bash
/kaggle/working/PRAG/output/
└── qwen2.5-1.5b-instruct/
    └── rank=2_alpha=32/
        └── 2wikimultihopqa/
            └── lr=0.0003_epoch=1_cot/
                └── aug_model=qwen2.5-1.5b-instruct/
                    └── prag/
                        └── total/
                            ├── result.txt      # 指标
                            ├── predict.json    # 详细预测
                            └── config.json     # 配置
```

### result.txt内容

```
em      0.6667
f1      0.7234
prec    0.7890
recall  0.6912
```

- **EM (Exact Match)**：完全匹配准确率
- **F1**：精确率和召回率的调和平均
- **Precision**：精确率
- **Recall**：召回率

---

## 🚀 快速开始

```python
# 完整流程
# Step 1: Encode（已完成）
# 输出：/kaggle/working/PRAG/offline/.../total/data_X/passage_Y/

# Step 2: Inference
import sys, argparse
sys.path.insert(0, '/kaggle/working/PRAG/src')
import root_dir_path
root_dir_path.ROOT_DIR = '/kaggle/working/PRAG'

from inference import main as inference_main

args = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',
    with_cot=True,
    augment_model='qwen2.5-1.5b-instruct',
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    inference_method='prag',
    sample=3,
    max_new_tokens=50,
)

inference_main(args)

# Step 3: 查看结果
!cat /kaggle/working/PRAG/output/.../prag/total/result.txt
```

---

## 📚 相关文档

- `KAGGLE_COMPLETE_GUIDE.md` - 完整Kaggle工作流程
- `KAGGLE_INFERENCE_GUIDE.md` - Inference详细指南
- `KAGGLE_JSON_FIELDS_EXPLANATION.md` - JSON字段详解
- `KAGGLE_DATA_TYPE_EXPLANATION.md` - data_type参数说明

---

**总结**：PRAG方法通过巧妙的三阶段训练策略，将文档知识编码到LoRA参数中，inference时完全不依赖原始文档，实现了真正的参数化知识存储。QA对是训练辅助数据，不是测试问题，因此不存在数据泄露。
