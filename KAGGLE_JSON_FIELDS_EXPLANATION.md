# Kaggle数据文件字段详解 / JSON Fields Explanation

本文档详细解释PRAG项目中JSON数据文件的所有字段含义和作用。

---

## 问题1：golden_passages字段

### 字段含义

`golden_passages` = **标准答案文档/金标准文档**

这个字段存储了**原始数据集中标注的、包含正确答案的参考文档**。

### 作用和用途

1. **数据增强的基准**
   - 在数据增强阶段（augment.py），从golden_passages提取关键信息
   - 用于生成训练数据的QA对

2. **评估和验证**
   - 检查检索到的文档质量（对比retrieved passages vs golden_passages）
   - 验证模型是否学习到了正确的知识

3. **训练数据构造**
   - encode.py使用golden_passages中的文档进行重写（rewrite）
   - 生成QA对训练LoRA参数

### 示例解读

```json
{
    "qid": "8813f87c0bdd11eba7f7acde48001122",
    "question": "Who is the mother of the director of film Polish-Russian War (Film)?",
    "answer": ["Małgorzata Braunek"],
    "golden_passages": [
        "Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski...",
        "Xawery Żuławski (born 22 December 1971 in Warsaw) is a Polish film director... His father is actress Małgorzata Braunek..."
    ],
    "type": "compositional"
}
```

**解读**：
- 问题需要两步推理：①找到电影导演 ②找到导演的母亲
- `golden_passages[0]`：包含"导演是Xawery Żuławski"
- `golden_passages[1]`：包含"母亲是Małgorzata Braunek"
- 两个文档组合才能回答问题

---

## 问题2：total.json的索引作用

### total.json的真实结构

**纠正之前的说明**：total.json **不是索引文件**，而是**元数据精简版**！

```json
// total.json内容（500KB）
{
    "qid": "xxx",
    "question": "...",
    "answer": [...],
    "golden_passages": [...],  // ✅ 包含golden_passages
    "type": "compositional"     // ✅ 问题类型标签
}
```

```json
// compositional.json内容（3000KB）
{
    "qid": "xxx",
    "question": "...",
    "answer": [...],
    "golden_passages": [...],
    "type": "compositional",
    "augment": [  // ❌ total.json没有这个字段！
        {
            "pid": 0,
            "passage": "retrieved document...",
            "qwen2.5-1.5b-instruct_rewrite": "rewritten version...",
            "qwen2.5-1.5b-instruct_qa": [
                {"question": "...", "answer": "...", "full_answer": "..."},
                {"question": "...", "answer": "...", "full_answer": "..."},
                {"question": "...", "answer": "...", "full_answer": "..."}
            ]
        },
        // ... 更多passages
    ],
    "passages": ["passage1", "passage2", ...]  // ❌ total.json没有这个字段
}
```

### 文件大小差异原因

| 文件 | 大小 | 包含内容 |
|------|------|---------|
| total.json | 500KB | 仅基础信息（question, answer, golden_passages, type） |
| compositional.json | 3000KB | **基础信息 + augment字段（重写文档+QA对）** |
| comparison.json | 3000KB | 基础信息 + augment字段 |
| inference.json | 3000KB | 基础信息 + augment字段 |
| bridge_comparison.json | 3000KB | 基础信息 + augment字段 |

### total.json的索引作用机制

查看`utils.py`第85-100行的代码：

```python
if data_type == "total":  # 合并所有类型
    all_data = {}
    for filename in files:
        all_data[filename] = json.load(fin)
    
    total_data = []
    idx = {filename: 0 for filename in files}
    
    for data in all_data["total.json"]:  # ← 遍历total.json
        typ = data["type"] + ".json"      # ← 获取类型标签
        aim_data = all_data[typ][idx[typ]]  # ← 从对应类型文件读取完整数据
        assert aim_data["question"] == data["question"]  # ← 验证匹配
        idx[typ] += 1
        total_data.append(aim_data)  # ← 使用类型文件的完整数据
```

**索引作用**：
1. total.json通过`type`字段标记每个问题属于哪个类型
2. 代码按total.json的顺序，从对应类型文件中读取**完整的augment数据**
3. 这样可以保持数据顺序一致，同时使用类型文件的完整增强信息

### 是否需要同时存在？

**是的！** 当使用`data_type="total"`时：

```python
# ✅ 正确：两类文件都需要
data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/
├── total.json              # 索引文件（按顺序列出所有问题+类型标签）
├── compositional.json      # 包含compositional类型的完整augment数据
├── comparison.json         # 包含comparison类型的完整augment数据
├── inference.json          # 包含inference类型的完整augment数据
└── bridge_comparison.json  # 包含bridge_comparison类型的完整augment数据
```

**工作流程**：
1. 读取total.json获取问题列表和类型标签
2. 根据type字段从相应的类型文件读取augment数据
3. 合并为一个统一数据集进行训练

---

## 问题3：model_name_translate字段

您看到的是`qwen2.5-1.5b-instruct_translate`字段（**注意是translate不是rewrite**）。

### 这个字段不在标准数据中

从代码分析（augment.py和encode.py），标准的增强数据字段是：

```python
{
    "pid": 0,
    "passage": "原始检索到的文档",
    f"{model_name}_rewrite": "重写后的文档版本",  # ← 只有rewrite
    f"{model_name}_qa": [...]  # QA对
}
```

### 可能的原因

1. **自定义扩展**：您的数据可能包含了额外的翻译功能
2. **不同版本的augment.py**：可能有translate功能的分支
3. **手动添加的字段**：用于多语言实验

### 对复现的影响

**不影响复现**！encode.py只使用这三个字段：
- `passage`：原始文档
- `{model_name}_rewrite`：重写版本
- `{model_name}_qa`：QA对

其他字段（如translate）会被**忽略**。

---

## 数据使用策略建议

### 策略1：使用data_type="total"（推荐用于论文复现）

```python
args_encode = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',  # ← 使用total模式
    # ...
)
```

**优点**：
- 一次训练处理所有问题类型
- 与论文实验设置一致
- 输出单个LoRA集合

**需要文件**：total.json + 所有类型文件

---

### 策略2：使用data_type=None（当前您的设置）

```python
args_encode = argparse.Namespace(
    data_type=None,  # ← 分别处理各类型
    # ...
)
```

**优点**：
- 分别为每个类型训练LoRA
- 可以分析不同类型的效果差异
- 便于调试单个类型

**缺点**：
- 需要运行多次（compositional, comparison, inference, bridge_comparison）
- 无法在一个模型中综合所有类型知识

**输出示例**（您的运行结果）：
```
### Solving inference ###
### Solving compositional ###
### Solving comparison ###
### Solving bridge_comparison ###
```

---

### 策略3：使用data_type="comparison"（单一类型）

```python
args_encode = argparse.Namespace(
    data_type='comparison',  # ← 只处理comparison类型
    # ...
)
```

**用途**：针对性实验、消融研究

---

## 文件必需性总结

### 最小文件集（data_type=None或特定类型）

```
PRAG/
├── src/（所有.py文件）
└── data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/
    ├── compositional.json   ✅ 必需（包含augment）
    ├── comparison.json      ✅ 必需
    ├── inference.json       ✅ 必需
    ├── bridge_comparison.json ✅ 必需
    └── total.json           ❌ 可选（data_type=None时不使用）
```

### 完整文件集（data_type="total"）

```
PRAG/
├── src/
└── data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/
    ├── total.json           ✅ 必需（索引文件）
    ├── compositional.json   ✅ 必需（augment数据）
    ├── comparison.json      ✅ 必需
    ├── inference.json       ✅ 必需
    └── bridge_comparison.json ✅ 必需
```

---

## 快速诊断脚本

```python
import json
import os

# 检查JSON文件内容
dataset_dir = '/kaggle/working/PRAG/data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct'

for filename in ['total.json', 'compositional.json']:
    filepath = os.path.join(dataset_dir, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"\n=== {filename} ===")
    print(f"数据条数: {len(data)}")
    print(f"文件大小: {os.path.getsize(filepath) / 1024:.1f} KB")
    
    # 检查第一条数据的字段
    first_item = data[0]
    print(f"字段列表: {list(first_item.keys())}")
    
    # 检查是否有augment字段
    if 'augment' in first_item:
        print(f"  ✅ 包含augment字段（增强数据）")
        print(f"  augment条数: {len(first_item['augment'])}")
        if first_item['augment']:
            aug_fields = list(first_item['augment'][0].keys())
            print(f"  augment字段: {aug_fields}")
    else:
        print(f"  ❌ 不包含augment字段（仅元数据）")
    
    if 'golden_passages' in first_item:
        print(f"  ✅ 包含golden_passages: {len(first_item['golden_passages'])}条")
```

**预期输出**：
```
=== total.json ===
数据条数: 100
文件大小: 500.2 KB
字段列表: ['qid', 'test_id', 'question', 'answer', 'golden_passages', 'type']
  ❌ 不包含augment字段（仅元数据）
  ✅ 包含golden_passages: 2条

=== compositional.json ===
数据条数: 50
文件大小: 3024.8 KB
字段列表: ['qid', 'test_id', 'question', 'answer', 'golden_passages', 'type', 'augment', 'passages']
  ✅ 包含augment字段（增强数据）
  augment条数: 3
  augment字段: ['pid', 'passage', 'qwen2.5-1.5b-instruct_rewrite', 'qwen2.5-1.5b-instruct_qa']
  ✅ 包含golden_passages: 2条
```

---

## 常见问题

### Q1: 为什么我的运行跳过了total.json？

**A**: 因为`data_type=None`时，代码会跳过total.json（utils.py第102行）。这是**正常行为**，不是错误！

### Q2: 我应该使用哪种data_type设置？

**A**: 
- **论文复现**：使用`data_type="total"`
- **调试/实验**：使用`data_type=None`或特定类型
- **快速测试**：使用`data_type="comparison"`（单一类型，数据量小）

### Q3: golden_passages会被训练使用吗？

**A**: **不会直接使用**。训练使用的是`augment`字段中的passages、rewrite和qa。golden_passages仅用于：
- 数据增强时的参考
- 评估和分析

### Q4: translate字段有什么用？

**A**: 不在标准流程中，可能是自定义扩展。**可以忽略**，不影响encode和inference。

---

## 推荐配置

### 完整论文复现

```python
# Cell: 准备环境
WORK_DIR = '/kaggle/working/PRAG'
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'

# 复制所有JSON文件
!mkdir -p {WORK_DIR}/data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct
!cp {INPUT_BASE}/data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/*.json \
   {WORK_DIR}/data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/

# 验证
import os
for fname in ['total.json', 'compositional.json', 'comparison.json', 'inference.json', 'bridge_comparison.json']:
    path = f'{WORK_DIR}/data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct/{fname}'
    exists = os.path.exists(path)
    size = os.path.getsize(path) / 1024 if exists else 0
    print(f"{'✅' if exists else '❌'} {fname}: {size:.1f} KB")
```

```python
# Cell: 运行encode
args_encode = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',  # ← 使用total模式合并所有类型
    with_cot=True,
    sample=10,
    augment_model='qwen2.5-1.5b-instruct',
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32
)

encode_main(args_encode)
```

**预期输出**：
```
### Solving total ###  ← 注意：只有一个"total"而非四个类型
100%|██████████| 10/10 [01:30<00:00, 9.0s/it]
✅ Encode完成！
```

---

**相关文档**：
- `KAGGLE_DATA_TYPE_EXPLANATION.md` - data_type参数详解
- `KAGGLE_COMPLETE_GUIDE.md` - 完整运行流程
- `src/utils.py` 第78-119行 - load_data函数实现
- `src/augment.py` 第60-85行 - golden_passages生成逻辑
