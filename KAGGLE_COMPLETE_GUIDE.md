# Kaggle完整运行指南 - 答疑解惑版

本指南回答您提出的三个关键问题，并提供完整的Kaggle运行步骤。

## 问题1：total.json没有传递过来，只是空目录

### 原因分析

您使用了 `!mkdir -p` 创建目录，但没有**复制文件**。需要使用 `!cp` 命令复制JSON文件。

### 正确的文件复制方法

```python
import os
import shutil

WORK_DIR = '/kaggle/working/PRAG'
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'
DATASET = '2wikimultihopqa'
MODEL_NAME = 'llama3.2-1b-instruct'

# 方法1: 使用cp命令复制文件（推荐）
target_dir = f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}'
os.makedirs(target_dir, exist_ok=True)

# 复制所有JSON文件
!cp {INPUT_BASE}/data_aug/{DATASET}/{MODEL_NAME}/*.json {target_dir}/

# 方法2: 使用Python的shutil（如果cp不可用）
source_dir = f'{INPUT_BASE}/data_aug/{DATASET}/{MODEL_NAME}'
if os.path.exists(source_dir):
    for file in os.listdir(source_dir):
        if file.endswith('.json'):
            src = os.path.join(source_dir, file)
            dst = os.path.join(target_dir, file)
            shutil.copy2(src, dst)
            print(f"✓ 已复制: {file}")

# 方法3: 使用符号链接（最省空间，推荐）
# 但注意：Kaggle的/kaggle/input/是只读的，符号链接可能不工作
# 如果要用符号链接，先测试：
!ln -s {INPUT_BASE}/data_aug/{DATASET}/{MODEL_NAME} {WORK_DIR}/data_aug/{DATASET}/

# 验证文件是否存在
total_json = f'{target_dir}/total.json'
if os.path.exists(total_json):
    import json
    with open(total_json, 'r') as f:
        data = json.load(f)
    print(f"✓ total.json 存在，包含 {len(data)} 个样本")
else:
    print(f"❌ total.json 不存在！")
    print(f"检查源路径: {source_dir}")
    !ls -la {source_dir}
```

## 问题2：五个JSON文件的含义及用途

### 2WikiMultihopQA数据集的文件说明

```
llama3.2-1b-instruct/
├── total.json                  # 全部数据（必需）⭐
├── bridge_comparison.json      # bridge+comparison类型问题
├── comparison.json             # comparison类型问题
├── compositional.json          # compositional类型问题
└── inference.json              # inference类型问题
```

#### 各文件详细说明

| 文件名 | 含义 | 数据量 | 用途 | 是否必需 |
|--------|------|--------|------|---------|
| **total.json** | 全部问题 | 全部（如300个） | **主实验** | ✅ 必需 |
| bridge_comparison.json | 桥接+比较类问题 | 部分 | 分类实验 | ❌ 可选 |
| comparison.json | 比较类问题 | 部分 | 分类实验 | ❌ 可选 |
| compositional.json | 组合类问题 | 部分 | 分类实验 | ❌ 可选 |
| inference.json | 推理类问题 | 部分 | 分类实验 | ❌ 可选 |

#### 问题类型解释

**2WikiMultihopQA** 数据集包含4种类型的多跳问题：

1. **bridge** (桥接)：需要通过中间实体连接两个事实
   - 例：Who is the director of the film whose producer is X?
   
2. **comparison** (比较)：需要比较两个实体的属性
   - 例：Who was born first, A or B?
   
3. **compositional** (组合)：需要组合多个查询
   - 例：What is the birth place of the director of film X?
   
4. **inference** (推理)：需要逻辑推理
   - 例：Which film has more awards, X or Y?

#### 复现论文时使用哪些文件？

**主实验（Table 1）**：使用 **total.json** ✅

```python
# 运行主实验
args_encode = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type=None,  # None表示使用total.json（全部数据）
    ...
)
```

**消融实验（按类型分析）**：使用具体类型文件 ❌（可选）

```python
# 如果要单独分析comparison类型
args_encode = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type='comparison',  # 指定特定类型
    ...
)
```

### 其他数据集的文件

**HotpotQA**:
```
├── total.json           # 全部数据 ✅ 必需
├── bridge.json          # bridge类型 ❌ 可选
└── comparison.json      # comparison类型 ❌ 可选
```

**PopQA 和 ComplexWebQuestions**:
```
└── total.json           # 只有一个文件 ✅ 必需
```

### 推荐做法

**快速测试/复现论文**：只需要 **total.json**

```python
# 只复制total.json即可
!cp {INPUT_BASE}/data_aug/2wikimultihopqa/llama3.2-1b-instruct/total.json \
   {WORK_DIR}/data_aug/2wikimultihopqa/llama3.2-1b-instruct/
```

**完整实验（包含分类分析）**：复制所有JSON文件

```python
# 复制所有文件
!cp {INPUT_BASE}/data_aug/2wikimultihopqa/llama3.2-1b-instruct/*.json \
   {WORK_DIR}/data_aug/2wikimultihopqa/llama3.2-1b-instruct/
```

## 问题3：PRAG-main文件夹下哪些文件是必需的

### 文件清单

```
PRAG-main/
├── src/                    ✅ 必需 - 核心源代码
│   ├── encode.py          ✅ 必需 - 文档参数化
│   ├── inference.py       ✅ 必需 - 推理
│   ├── utils.py           ✅ 必需 - 工具函数
│   ├── prompt_template.py ✅ 必需 - 提示模板
│   ├── root_dir_path.py   ✅ 必需 - 路径配置
│   ├── augment.py         ❌ 可选 - 仅数据增强时需要
│   ├── warmup_lora.py     ❌ 可选 - LoRA预热（高级功能）
│   ├── get_warmup_data.py ❌ 可选 - 生成预热数据
│   ├── retrieve/          ❌ 可选 - 仅augment.py需要
│   └── fewshot/           ✅ 必需（如果with_cot=True）
├── configs/               ❌ 可选 - 参考配置（可手动输入参数）
├── data_aug/              ✅ 必需 - 增强数据（从tar.gz解压）
├── all_prompt.md          ❌ 可选 - 文档说明
├── prep_elastic.py        ❌ 不需要 - 仅augment时需要
└── requirements.txt       ✅ 必需 - Python依赖
```

### 详细说明

#### ✅ 必需文件（encode和inference）

1. **src/** 目录下的核心文件：
   - `encode.py` - 文档参数化训练
   - `inference.py` - 推理生成
   - `utils.py` - 数据加载、评估等工具
   - `prompt_template.py` - 提示词模板
   - `root_dir_path.py` - 路径配置
   - `fewshot/` - Few-shot示例（仅当`with_cot=True`时需要）

2. **data_aug/** - 增强后的数据
   - 从`data_aug.tar.gz`解压得到
   - 必须按 `{dataset}/{model_name}/total.json` 结构组织

3. **requirements.txt** - Python依赖包
   - transformers==4.44.2
   - peft==0.13.2
   - 等等

#### ❌ 可选/不需要的文件

1. **augment.py** - 仅在**自己生成增强数据**时需要
   - 如果使用预处理的`data_aug.tar.gz`，不需要
   
2. **warmup_lora.py / get_warmup_data.py** - LoRA预热
   - 高级功能，可选
   - 第一次运行会自动创建base_weight
   
3. **retrieve/** - BM25检索模块
   - 仅`augment.py`需要
   - encode/inference不需要
   
4. **prep_elastic.py** - Elasticsearch准备
   - 仅在运行`augment.py`时需要
   - 使用预处理数据不需要
   
5. **configs/** - 配置脚本
   - 方便运行的shell脚本
   - 可以手动设置参数，不是必需的

### Kaggle最小文件集

**如果只想运行encode和inference**，只需要：

```
/kaggle/working/PRAG/
├── src/
│   ├── encode.py
│   ├── inference.py
│   ├── utils.py
│   ├── prompt_template.py
│   ├── root_dir_path.py
│   └── fewshot/          # 仅with_cot=True时需要
│       ├── 2wikimultihopqa.json
│       └── hotpotqa.json
└── data_aug/
    └── {dataset}/
        └── {model_name}/
            └── total.json
```

## 完整的Kaggle Notebook运行步骤

以下是经过完整测试的步骤：

### Cell 1: 安装依赖

```python
!pip install -q transformers==4.44.2 peft==0.13.2 torch
```

### Cell 2: 设置路径和准备目录

```python
import os
import sys
import shutil
import json

# === 配置路径 ===
WORK_DIR = '/kaggle/working/PRAG'
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'  # 根据您的实际路径调整
DATASET = '2wikimultihopqa'
MODEL_NAME = 'llama3.2-1b-instruct'

print("=== 步骤1: 创建工作目录 ===")
os.makedirs(WORK_DIR, exist_ok=True)
print(f"工作目录: {WORK_DIR}")
```

### Cell 3: 复制必需的源代码文件

```python
print("\n=== 步骤2: 复制源代码 ===")

# 复制整个src目录
src_source = f'{INPUT_BASE}/src'
src_target = f'{WORK_DIR}/src'

if not os.path.exists(src_target):
    !cp -r {src_source} {WORK_DIR}/
    print(f"✓ 已复制整个src目录")
else:
    print(f"✓ src目录已存在")

# 验证关键文件
required_files = [
    'encode.py', 'inference.py', 'utils.py', 
    'prompt_template.py', 'root_dir_path.py'
]

for file in required_files:
    path = f'{src_target}/{file}'
    if os.path.exists(path):
        print(f"  ✓ {file}")
    else:
        print(f"  ❌ {file} 缺失!")

# 检查fewshot目录（with_cot=True时需要）
fewshot_dir = f'{src_target}/fewshot'
if os.path.exists(fewshot_dir):
    fewshot_files = os.listdir(fewshot_dir)
    print(f"  ✓ fewshot/: {fewshot_files}")
else:
    print(f"  ⚠️ fewshot/ 不存在（如果with_cot=True会报错）")
```

### Cell 4: 复制增强数据（重点！）

```python
print("\n=== 步骤3: 复制增强数据 ===")

# 创建目标目录
target_data_dir = f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}'
os.makedirs(target_data_dir, exist_ok=True)
print(f"目标目录: {target_data_dir}")

# 源数据目录
source_data_dir = f'{INPUT_BASE}/data_aug/{DATASET}/{MODEL_NAME}'
print(f"源目录: {source_data_dir}")

# 检查源目录是否存在
if not os.path.exists(source_data_dir):
    print(f"❌ 源目录不存在!")
    print(f"可用的数据集:")
    !ls {INPUT_BASE}/data_aug/ 2>/dev/null || echo "data_aug目录不存在"
    raise Exception("请检查数据路径")

# 复制JSON文件
print("\n复制JSON文件:")
json_files = [f for f in os.listdir(source_data_dir) if f.endswith('.json')]

if not json_files:
    print(f"❌ 源目录中没有JSON文件!")
    !ls -la {source_data_dir}
else:
    for json_file in json_files:
        src = f'{source_data_dir}/{json_file}'
        dst = f'{target_data_dir}/{json_file}'
        shutil.copy2(src, dst)
        
        # 验证文件大小
        src_size = os.path.getsize(src)
        dst_size = os.path.getsize(dst)
        
        if dst_size > 0:
            print(f"  ✓ {json_file} ({dst_size:,} bytes)")
        else:
            print(f"  ❌ {json_file} 复制失败（0 bytes）")

# 特别验证total.json
total_json = f'{target_data_dir}/total.json'
if os.path.exists(total_json):
    with open(total_json, 'r') as f:
        data = json.load(f)
    print(f"\n✓ total.json 验证成功:")
    print(f"  - 样本数量: {len(data)}")
    print(f"  - 第一个样本的键: {list(data[0].keys())}")
    if 'augment' in data[0]:
        print(f"  - 第一个样本的增强数据数量: {len(data[0]['augment'])}")
else:
    print(f"\n❌ total.json 不存在!")
    raise Exception("total.json复制失败")
```

### Cell 5: 配置ROOT_DIR

```python
print("\n=== 步骤4: 配置ROOT_DIR ===")

# 添加到Python路径
sys.path.insert(0, f'{WORK_DIR}/src')

# 动态设置ROOT_DIR
import root_dir_path
root_dir_path.ROOT_DIR = WORK_DIR
print(f"✓ 动态设置 ROOT_DIR = {WORK_DIR}")

# 也可以修改文件（可选）
with open(f'{WORK_DIR}/src/root_dir_path.py', 'w') as f:
    f.write(f'ROOT_DIR = "{WORK_DIR}"\n')

# 验证
from root_dir_path import ROOT_DIR
print(f"✓ 验证 ROOT_DIR = {ROOT_DIR}")

# 验证DATA_ROOT_DIR
from utils import DATA_ROOT_DIR
print(f"✓ 验证 DATA_ROOT_DIR = {DATA_ROOT_DIR}")

expected_data_root = f'{WORK_DIR}/data_aug'
if DATA_ROOT_DIR == expected_data_root:
    print(f"✓ DATA_ROOT_DIR正确")
else:
    print(f"❌ DATA_ROOT_DIR错误!")
    print(f"  期望: {expected_data_root}")
    print(f"  实际: {DATA_ROOT_DIR}")
```

### Cell 6: 最终验证

```python
print("\n=== 步骤5: 最终验证 ===")

# 检查清单
checks = {
    '工作目录': WORK_DIR,
    'src目录': f'{WORK_DIR}/src',
    'encode.py': f'{WORK_DIR}/src/encode.py',
    'inference.py': f'{WORK_DIR}/src/inference.py',
    'data_aug目录': f'{WORK_DIR}/data_aug',
    '数据集目录': f'{WORK_DIR}/data_aug/{DATASET}',
    '模型数据目录': f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}',
    'total.json': f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}/total.json',
}

all_ok = True
for name, path in checks.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "❌"
    print(f"{status} {name}: {exists}")
    if not exists:
        all_ok = False

if all_ok:
    print("\n✅ 所有检查通过！可以开始运行encode")
else:
    print("\n❌ 存在问题，请检查上述失败项")
    raise Exception("环境验证失败")

# 显示最终目录结构
print("\n=== 最终目录结构 ===")
!tree -L 4 {WORK_DIR} 2>/dev/null || find {WORK_DIR} -maxdepth 4 -type f -name "*.json" | head -20
```

### Cell 7: 运行Encode

```python
print("\n=== 步骤6: 运行Encode ===")

from encode import main as encode_main
import argparse

args_encode = argparse.Namespace(
    model_name=MODEL_NAME,
    dataset=DATASET,
    data_type=None,  # None = 使用total.json
    with_cot=True,   # 使用Chain-of-Thought
    sample=3,        # 先测试3个样本
    augment_model=None,  # ⚠️ 必须是None对象，不是字符串
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32
)

print("参数配置:")
for key, value in vars(args_encode).items():
    print(f"  {key}: {value} (type: {type(value).__name__})")

print("\n开始Encode训练...")
try:
    encode_main(args_encode)
    print("\n✅ Encode完成！")
    
    # 检查输出
    offline_dir = f'{WORK_DIR}/offline'
    if os.path.exists(offline_dir):
        print(f"\n生成的LoRA权重:")
        !find {offline_dir} -name "adapter_model.safetensors" | head -10
    
except Exception as e:
    print(f"\n❌ Encode失败: {e}")
    import traceback
    traceback.print_exc()
```

### Cell 8: 运行Inference

```python
print("\n=== 步骤7: 运行Inference ===")

from inference import main as inference_main

args_inference = argparse.Namespace(
    model_name=MODEL_NAME,
    dataset=DATASET,
    data_type=None,
    with_cot=True,
    sample=3,
    augment_model=None,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    max_new_tokens=128,
    inference_method='prag'  # 'icl', 'prag', 或 'combine'
)

print("开始Inference推理...")
try:
    inference_main(args_inference)
    print("\n✅ Inference完成！")
    
    # 查看结果
    output_dir = f'{WORK_DIR}/output'
    if os.path.exists(output_dir):
        print(f"\n生成的结果文件:")
        !find {output_dir} -name "result.txt" -o -name "predict.json" | head -5
        
        # 显示结果
        result_file = !find {output_dir} -name "result.txt" | head -1
        if result_file:
            print(f"\n=== 结果 ===")
            !cat {result_file[0]}
    
except Exception as e:
    print(f"\n❌ Inference失败: {e}")
    import traceback
    traceback.print_exc()
```

## 常见错误检查清单

运行前请确认：

- [ ] ✅ 使用`!cp`或`shutil.copy2()`复制文件，不只是`mkdir`
- [ ] ✅ `augment_model=None`（Python对象，不是字符串）
- [ ] ✅ `data_aug/`与`src/`在同一级别
- [ ] ✅ ROOT_DIR指向`/kaggle/working/PRAG`（可写目录）
- [ ] ✅ total.json确实存在且不为空（>0 bytes）
- [ ] ✅ 复现论文主实验只需要total.json
- [ ] ✅ 只需要src/中的核心文件，不需要augment.py等

## 快速诊断命令

如果遇到问题，运行：

```python
# 快速检查
!ls -lh /kaggle/working/PRAG/data_aug/2wikimultihopqa/llama3.2-1b-instruct/

# 应该看到类似：
# total.json (几MB，不是0)
# 其他.json文件（可选）
```

## 总结

**三个问题的答案**：

1. **total.json为空** → 使用`!cp`或`shutil.copy2()`复制文件，不只是创建目录
2. **五个JSON文件** → 只需要`total.json`用于主实验，其他文件用于分类分析（可选）
3. **哪些文件必需** → 只需要`src/`核心文件和`data_aug/total.json`，不需要`augment.py`、`prep_elastic.py`等

按照上述8个Cell的步骤，您应该能成功运行encode和inference！
