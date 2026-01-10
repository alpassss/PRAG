# Kaggle Inference完全指南

## 问题：LoRA权重路径是否正确？如何运行Inference？

### 一、LoRA权重路径解析

您生成的路径**完全正确**！✅

```
/kaggle/working/PRAG/offline/qwen2.5-1.5b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=qwen2.5-1.5b-instruct/total/data_0/passage_0/adapter_model.safetensors
```

#### 路径结构说明

```
offline/
├── {model_name}/                    # qwen2.5-1.5b-instruct
│   ├── base_weight/                 # 基础LoRA权重（用于初始化）
│   │   └── adapter_model.safetensors
│   └── rank={r}_alpha={a}/          # rank=2_alpha=32
│       └── {dataset}/               # 2wikimultihopqa
│           └── lr={lr}_epoch={e}_{cot}/  # lr=0.0003_epoch=1_cot
│               └── aug_model={model}/    # aug_model=qwen2.5-1.5b-instruct
│                   └── {data_type}/      # total (或 comparison, compositional等)
│                       └── data_{i}/     # 数据样本ID (data_0, data_1, ...)
│                           ├── passage_0/   # 每个文档的LoRA权重
│                           ├── passage_1/
│                           └── passage_2/
│                               └── adapter_model.safetensors
```

#### 关键点

1. **每个文档有独立的LoRA权重**：
   - `data_0/passage_0/` = 第1个问题的第1个文档的权重
   - `data_0/passage_1/` = 第1个问题的第2个文档的权重
   - `data_1/passage_0/` = 第2个问题的第1个文档的权重

2. **base_weight**：所有LoRA的起始权重，推理时不直接使用

3. **data_type路径**：您使用`data_type='total'`，所以路径包含`/total/`

---

## 二、Inference完整流程

### 步骤1：准备测试数据

Inference需要**测试集数据**，与encode训练数据分开。

```python
# Cell 7: 下载测试数据 (如果还没有)
import os

# 方式1：如果测试数据已在data_aug中
TEST_DATA_PATH = '/kaggle/working/PRAG/data_aug/2wikimultihopqa/qwen2.5-1.5b-instruct'

# 方式2：如果需要单独下载测试集（通常从原始数据集获取）
# 注意：PRAG使用增强数据作为测试数据，应该已经在data_aug中
```

### 步骤2：运行Inference

在Kaggle Notebook中有**两种运行方式**：

#### **方式A：Python函数调用（推荐）**

```python
# Cell 8: 运行Inference
import sys
import argparse
sys.path.insert(0, '/kaggle/working/PRAG/src')

# 确保ROOT_DIR正确
import root_dir_path
root_dir_path.ROOT_DIR = '/kaggle/working/PRAG'

from inference import main as inference_main

# 配置参数
args_inference = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',  # 必须与encode时一致
    with_cot=True,      # 必须与encode时一致
    sample=3,           # 测试前3个样本，-1表示全部
    augment_model='qwen2.5-1.5b-instruct',  # 必须与encode时一致
    
    # 训练参数（必须与encode时完全一致以找到正确的权重路径）
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    
    # Inference特有参数
    inference_method='prag',  # 三个选项：'icl', 'prag', 'combine'
    max_new_tokens=50,        # 生成答案的最大token数
)

print("=== Inference参数配置 ===")
for key, value in vars(args_inference).items():
    print(f"  {key}: {value} (type: {type(value).__name__})")

print("\n开始Inference...")
try:
    inference_main(args_inference)
    print("\n✅ Inference完成！")
except Exception as e:
    print(f"\n❌ Inference失败: {e}")
    import traceback
    traceback.print_exc()
```

#### **方式B：命令行模式**

```python
# Cell 8备选：命令行运行
!cd /kaggle/working/PRAG && python src/inference.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset 2wikimultihopqa \
    --data_type total \
    --with_cot \
    --sample 3 \
    --augment_model qwen2.5-1.5b-instruct \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --inference_method prag \
    --max_new_tokens 50
```

---

## 三、Inference方法详解

### 三种inference_method对比

| 方法 | 说明 | 使用LoRA | 使用文档 | 适用场景 |
|------|------|----------|----------|----------|
| **icl** | In-Context Learning | ❌ 否 | ✅ 是 | 基线对比：仅用原始模型+文档 |
| **prag** | PRAG方法 | ✅ 是 | ❌ 否 | 论文核心：仅用参数化知识 |
| **combine** | 组合方法 | ✅ 是 | ✅ 是 | 完整版：参数化知识+文档 |

#### 代码逻辑（inference.py第75-97行）

```python
if args.inference_method == "icl":
    # 方法1：ICL - 直接用原始模型+文档回答
    prediction = model.generate(question + passages)
    
else:  # "prag" 或 "combine"
    # 加载并合并该问题所有文档的LoRA权重
    for passage_id in range(num_passages):
        adapter_path = f"offline/.../data_{test_id}/passage_{passage_id}"
        model.load_adapter(adapter_path)
    
    # 合并LoRA权重（concatenation）
    model.merge_adapters(combination_type="cat")
    
    if args.inference_method == "prag":
        # 方法2：PRAG - 仅用参数化知识
        prediction = model.generate(question)  # 不给文档
    else:  # "combine"
        # 方法3：Combine - 参数化知识+文档
        prediction = model.generate(question + passages)
```

### 推荐配置

**复现论文主实验**：
```python
inference_method='prag'  # 仅用参数化知识
```

**完整对比实验**（运行3次）：
```python
# 实验1：基线
inference_method='icl'

# 实验2：PRAG（论文方法）
inference_method='prag'

# 实验3：组合方法
inference_method='combine'
```

---

## 四、输出结果位置

### 结果保存路径

```
output/
└── {model_name}/                    # qwen2.5-1.5b-instruct
    └── rank={r}_alpha={a}/          # rank=2_alpha=32
        └── {dataset}/               # 2wikimultihopqa
            └── lr={lr}_epoch={e}_{cot}/    # lr=0.0003_epoch=1_cot
                └── aug_model={model}/      # aug_model=qwen2.5-1.5b-instruct
                    └── {inference_method}/ # prag / icl / combine
                        └── {data_type}/    # total
                            ├── predict.json  # 详细预测结果
                            ├── result.txt    # 评估指标
                            └── config.json   # 运行配置
```

### 示例路径

```
/kaggle/working/PRAG/output/qwen2.5-1.5b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=qwen2.5-1.5b-instruct/prag/total/
```

---

## 五、查看结果和指标

### Cell 9：读取并显示结果

```python
import json
import os

# 定义输出目录
OUTPUT_DIR = '/kaggle/working/PRAG/output/qwen2.5-1.5b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=qwen2.5-1.5b-instruct/prag/total'

# 1. 读取评估指标
print("=== 评估指标 ===")
with open(f'{OUTPUT_DIR}/result.txt', 'r') as f:
    print(f.read())

# 2. 读取详细预测（前3个样本）
print("\n=== 预测详情（前3个样本）===")
with open(f'{OUTPUT_DIR}/predict.json', 'r') as f:
    predictions = json.load(f)
    
for i, pred in enumerate(predictions[:3]):
    print(f"\n--- 样本 {i} ---")
    print(f"问题: {pred['question']}")
    print(f"金标准答案: {pred['answer']}")
    print(f"模型预测: {pred['text']}")
    print(f"EM: {pred['em']}, F1: {pred['f1']}")
```

### 指标说明

生成的`result.txt`包含4个指标：

| 指标 | 全称 | 说明 | 取值范围 |
|------|------|------|----------|
| **em** | Exact Match | 完全匹配率 | 0-1 |
| **f1** | F1 Score | 综合精确率和召回率 | 0-1 |
| **prec** | Precision | 精确率 | 0-1 |
| **recall** | Recall | 召回率 | 0-1 |

**示例输出**：
```
em      0.6667
f1      0.7891
prec    0.8123
recall  0.7701
```

---

## 六、完整Kaggle Notebook代码

### Cell 10：一键运行完整Inference流程

```python
import sys
import argparse
import json
import os

# 1. 配置环境
sys.path.insert(0, '/kaggle/working/PRAG/src')
import root_dir_path
root_dir_path.ROOT_DIR = '/kaggle/working/PRAG'

from inference import main as inference_main

# 2. 配置参数
args_inference = argparse.Namespace(
    # 模型和数据（必须与encode一致）
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',
    with_cot=True,
    augment_model='qwen2.5-1.5b-instruct',
    
    # 训练配置（必须与encode一致）
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    
    # Inference配置
    inference_method='prag',  # 或 'icl', 'combine'
    sample=3,  # 测试3个样本，-1表示全部
    max_new_tokens=50,
)

# 3. 运行Inference
print("=== 开始Inference ===")
try:
    inference_main(args_inference)
    print("\n✅ Inference完成！")
except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    
# 4. 显示结果
OUTPUT_DIR = f'/kaggle/working/PRAG/output/{args_inference.model_name}/rank={args_inference.lora_rank}_alpha={args_inference.lora_alpha}/{args_inference.dataset}/lr={args_inference.learning_rate}_epoch={args_inference.num_train_epochs}_{"cot" if args_inference.with_cot else "direct"}/aug_model={args_inference.augment_model}/{args_inference.inference_method}/{args_inference.data_type or "None"}'

if os.path.exists(f'{OUTPUT_DIR}/result.txt'):
    print("\n=== 评估指标 ===")
    with open(f'{OUTPUT_DIR}/result.txt', 'r') as f:
        print(f.read())
else:
    print(f"\n结果文件未找到，请检查路径: {OUTPUT_DIR}")
```

---

## 七、常见问题排查

### 问题1：找不到LoRA权重

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: '.../offline/.../data_0/passage_0'
```

**原因**：Inference参数与Encode不一致

**解决**：确保以下参数**完全相同**：
- `model_name`
- `dataset`
- `data_type`
- `with_cot`
- `augment_model`
- `num_train_epochs`
- `learning_rate`
- `lora_rank`
- `lora_alpha`

### 问题2：sample参数设置错误

**现象**：Inference尝试处理的样本数超过Encode训练的数量

**解决**：
```python
# 如果Encode时用sample=3，则Inference最多也只能用sample=3
# Encode
args_encode.sample = 3  # 训练了3个样本

# Inference
args_inference.sample = 3  # 最多测试3个
# 或
args_inference.sample = 1  # 测试更少也可以
```

### 问题3：data_type不匹配

**错误**：Encode用`data_type=None`，Inference用`data_type='total'`

**解决**：两者必须一致
```python
# Encode和Inference都用
data_type='total'  # 或都用 None
```

---

## 八、参数快速对照表

### Encode vs Inference 参数对应

| 参数 | Encode | Inference | 是否必须相同 |
|------|--------|-----------|--------------|
| model_name | ✅ | ✅ | ✅ 是 |
| dataset | ✅ | ✅ | ✅ 是 |
| data_type | ✅ | ✅ | ✅ 是 |
| with_cot | ✅ | ✅ | ✅ 是 |
| augment_model | ✅ | ✅ | ✅ 是 |
| sample | ✅ | ✅ | ❌ 否（Inf≤Enc） |
| per_device_train_batch_size | ✅ | ❌ | - |
| num_train_epochs | ✅ | ✅ | ✅ 是 |
| learning_rate | ✅ | ✅ | ✅ 是 |
| lora_rank | ✅ | ✅ | ✅ 是 |
| lora_alpha | ✅ | ✅ | ✅ 是 |
| **inference_method** | ❌ | ✅ | - |
| **max_new_tokens** | ❌ | ✅ | - |

---

## 九、论文复现推荐配置

### 最小可行实验（快速验证）

```python
# Encode (Cell 6)
args_encode = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',
    with_cot=True,
    sample=5,  # 仅5个样本
    augment_model='qwen2.5-1.5b-instruct',
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
)

# Inference (Cell 8)
args_inference = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='2wikimultihopqa',
    data_type='total',
    with_cot=True,
    sample=5,
    augment_model='qwen2.5-1.5b-instruct',
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    inference_method='prag',
    max_new_tokens=50,
)
```

### 完整论文复现（时间和资源充足）

```python
# 完整数据集
sample=-1  # 处理所有数据

# 三个实验：ICL vs PRAG vs Combine
for method in ['icl', 'prag', 'combine']:
    args_inference.inference_method = method
    inference_main(args_inference)
```

---

## 十、诊断脚本

### Cell 11：验证LoRA权重和输出目录

```python
import os
import glob

WORK_DIR = '/kaggle/working/PRAG'

print("=== LoRA权重检查 ===")
offline_path = f'{WORK_DIR}/offline/qwen2.5-1.5b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=qwen2.5-1.5b-instruct/total'
if os.path.exists(offline_path):
    adapters = glob.glob(f'{offline_path}/**/adapter_model.safetensors', recursive=True)
    print(f"✅ 找到 {len(adapters)} 个LoRA权重文件")
    print(f"示例路径:\n{adapters[0] if adapters else '无'}")
else:
    print(f"❌ LoRA权重目录不存在: {offline_path}")

print("\n=== 输出目录检查 ===")
output_path = f'{WORK_DIR}/output'
if os.path.exists(output_path):
    results = glob.glob(f'{output_path}/**/result.txt', recursive=True)
    print(f"✅ 找到 {len(results)} 个结果文件")
    for r in results:
        print(f"  {r}")
else:
    print(f"❌ 输出目录不存在: {output_path}")
```

---

## 总结

1. **您的LoRA路径完全正确** ✅
2. **Inference核心**：参数必须与Encode一致以找到正确的权重路径
3. **三种方法**：icl（基线）、prag（论文方法）、combine（组合）
4. **结果位置**：`output/.../result.txt`包含EM、F1等指标
5. **快速验证**：先用`sample=3`小规模测试，再扩展到全数据集

**下一步**：复制上面的Cell 8代码到您的Notebook，运行Inference即可获得评估指标！
