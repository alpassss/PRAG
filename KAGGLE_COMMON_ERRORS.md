# Kaggle 常见错误完全解决指南

## 错误1: IsADirectoryError

### 错误场景
```python
augment_model=''  # 空字符串
```
报错：`IsADirectoryError`

### 根本原因

**问题1：augment_model参数错误**

```python
# ❌ 错误 - 空字符串
augment_model=''

# ❌ 错误 - 字符串'None'
augment_model='None'

# ✅ 正确 - Python的None对象
augment_model=None
```

当 `augment_model=''` (空字符串) 时：
- `encode.py` 第199行检查 `if args.augment_model is None:` 
- 空字符串不是 `None`，所以不会被替换为 `model_name`
- `utils.py` 第80行：`os.path.join(DATA_ROOT_DIR, data_name, model_name)` 
- 空字符串作为路径组件导致 `IsADirectoryError`

**问题2：data_aug位置错误**

您提到把 `data_aug` 放在 `src/` 下，这是**错误的**！

正确的目录结构：
```
PRAG/                          ✅ 正确
├── src/
│   ├── encode.py
│   ├── utils.py
│   └── root_dir_path.py
└── data_aug/
    └── 2wikimultihopqa/
        └── llama3.2-1b-instruct/
            └── total.json

src/                           ❌ 错误 - data_aug不应该在这里
└── data_aug/
    └── ...
```

为什么？因为 `DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data_aug")`
- 如果 `ROOT_DIR = '/kaggle/working/PRAG'`
- 那么 `DATA_ROOT_DIR = '/kaggle/working/PRAG/data_aug'`
- **不是** `/kaggle/working/PRAG/src/data_aug`

### 完整解决方案

```python
# ===== Cell 1: 安装依赖 =====
!pip install -q transformers==4.44.2 peft==0.13.2

# ===== Cell 2: 准备正确的目录结构 =====
import os
import sys
import shutil

# 定义路径
WORK_DIR = '/kaggle/working/PRAG'
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'
DATASET = '2wikimultihopqa'
MODEL_NAME = 'llama3.2-1b-instruct'

print("=== 步骤1: 创建工作目录 ===")
os.makedirs(WORK_DIR, exist_ok=True)
print(f"工作目录: {WORK_DIR}")

# ===== 步骤2: 复制源代码（到PRAG/，不是PRAG/src/）=====
print("\n=== 步骤2: 复制源代码 ===")
src_dir = f'{WORK_DIR}/src'
if not os.path.exists(src_dir):
    !cp -r {INPUT_BASE}/src {WORK_DIR}/
    print(f"✓ 源代码已复制到: {src_dir}")
else:
    print(f"✓ 源代码已存在: {src_dir}")

# ===== 步骤3: 准备data_aug目录（与src同级，不是在src内）=====
print("\n=== 步骤3: 准备data_aug目录 ===")

# 目标路径（与src同级）
target_data_dir = f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}'
os.makedirs(target_data_dir, exist_ok=True)

# 检查源数据位置
source_dataset_dir = f'{INPUT_BASE}/data_aug/{DATASET}'

if os.path.exists(f'{source_dataset_dir}/total.json'):
    # 情况1: 数据直接在dataset目录下（缺少model_name层）
    print(f"检测到数据在: {source_dataset_dir}/")
    print("创建符号链接并添加model_name层级...")
    
    for file in os.listdir(source_dataset_dir):
        if file.endswith('.json'):
            src = f'{source_dataset_dir}/{file}'
            dst = f'{target_data_dir}/{file}'
            if not os.path.exists(dst):
                os.symlink(src, dst)
                print(f"  ✓ 链接: {file}")
                
elif os.path.exists(f'{source_dataset_dir}/{MODEL_NAME}/total.json'):
    # 情况2: 数据结构正确
    print(f"检测到正确的数据结构: {source_dataset_dir}/{MODEL_NAME}/")
    target_dataset_dir = f'{WORK_DIR}/data_aug/{DATASET}'
    os.makedirs(os.path.dirname(target_dataset_dir), exist_ok=True)
    
    if not os.path.exists(target_dataset_dir):
        os.symlink(source_dataset_dir, target_dataset_dir)
        print(f"✓ 已链接: {source_dataset_dir}/ -> {target_dataset_dir}/")
        
else:
    print(f"❌ 未找到数据！")
    print(f"检查路径: {source_dataset_dir}/")
    !ls -la {source_dataset_dir} 2>/dev/null || echo "路径不存在"

# ===== 步骤4: 验证目录结构 =====
print("\n=== 步骤4: 验证目录结构 ===")

# 检查关键路径
checks = {
    'ROOT_DIR': WORK_DIR,
    'src目录': f'{WORK_DIR}/src',
    'data_aug目录': f'{WORK_DIR}/data_aug',
    '数据集目录': f'{WORK_DIR}/data_aug/{DATASET}',
    '模型数据目录': f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}',
    'total.json': f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}/total.json',
}

all_ok = True
for name, path in checks.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "❌"
    print(f"{status} {name}: {path}")
    if not exists and 'total.json' not in name:
        all_ok = False

if not all_ok:
    print("\n❌ 目录结构不正确！")
    print("\n当前WORK_DIR内容:")
    !ls -R {WORK_DIR}/ | head -50
    raise Exception("请修正目录结构")
else:
    print("\n✓ 目录结构验证通过！")

# 显示最终结构
print("\n=== 最终目录结构 ===")
print(f"""
{WORK_DIR}/
├── src/
│   ├── encode.py
│   ├── inference.py
│   ├── utils.py
│   └── ...
└── data_aug/              ⬅️ 与src同级，不是在src内！
    └── {DATASET}/
        └── {MODEL_NAME}/
            └── total.json
""")

# ===== Cell 3: 配置ROOT_DIR =====
print("\n=== 步骤5: 配置ROOT_DIR ===")

# 添加到Python路径
sys.path.insert(0, f'{WORK_DIR}/src')

# 方法1: 动态设置（推荐）
import root_dir_path
root_dir_path.ROOT_DIR = WORK_DIR
print(f"✓ 动态设置 ROOT_DIR = {WORK_DIR}")

# 方法2: 修改文件（可选）
with open(f'{WORK_DIR}/src/root_dir_path.py', 'w') as f:
    f.write(f'ROOT_DIR = "{WORK_DIR}"\n')
print(f"✓ 已修改 root_dir_path.py")

# 验证
from root_dir_path import ROOT_DIR
print(f"✓ 验证 ROOT_DIR = {ROOT_DIR}")

# 验证DATA_ROOT_DIR
from utils import DATA_ROOT_DIR
print(f"✓ 验证 DATA_ROOT_DIR = {DATA_ROOT_DIR}")
assert DATA_ROOT_DIR == f'{WORK_DIR}/data_aug', f"DATA_ROOT_DIR错误: {DATA_ROOT_DIR}"

# ===== Cell 4: 验证数据可访问 =====
print("\n=== 步骤6: 验证数据可访问 ===")

import json
from utils import load_data

try:
    # 测试加载数据
    data_list = load_data(DATASET, None, MODEL_NAME)
    print(f"✓ 成功加载数据!")
    
    for filename, data in data_list:
        print(f"  - {filename}: {len(data)} 个样本")
        if len(data) > 0:
            print(f"    第一个样本的键: {list(data[0].keys())}")
            
except Exception as e:
    print(f"❌ 加载数据失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 调试信息
    print("\n调试信息:")
    print(f"ROOT_DIR = {ROOT_DIR}")
    print(f"DATA_ROOT_DIR = {DATA_ROOT_DIR}")
    print(f"期望路径: {DATA_ROOT_DIR}/{DATASET}/{MODEL_NAME}/")
    !ls -la {DATA_ROOT_DIR}/{DATASET}/{MODEL_NAME}/ 2>/dev/null || echo "路径不存在"

# ===== Cell 5: 运行Encode =====
print("\n=== 步骤7: 运行Encode ===")

from encode import main as encode_main
import argparse

args_encode = argparse.Namespace(
    model_name=MODEL_NAME,
    dataset=DATASET,
    data_type=None,
    with_cot=True,
    sample=3,  # 先测试3个样本
    augment_model=None,  # ✅✅✅ 关键：必须是None，不是''或'None'
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
    print("\n✓ Encode完成！")
    
    # 检查输出
    offline_dir = f'{WORK_DIR}/offline'
    if os.path.exists(offline_dir):
        print(f"\n生成的LoRA权重:")
        !find {offline_dir} -name "adapter_model.safetensors" | head -5
    
except Exception as e:
    print(f"\n❌ Encode失败: {e}")
    import traceback
    traceback.print_exc()
```

## 错误对比表

| 参数设置 | 结果 | 原因 |
|---------|------|------|
| `augment_model=None` | ✅ 正确 | `is None` 检查通过，自动设为`model_name` |
| `augment_model=''` | ❌ IsADirectoryError | 空字符串不是None，作为路径组件失败 |
| `augment_model='None'` | ❌ FileNotFoundError | 字符串'None'不是None，找不到名为'None'的目录 |
| `augment_model='llama3.2-1b-instruct'` | ✅ 正确 | 明确指定模型名 |

## 目录结构对比

### ❌ 错误的结构（data_aug在src内）

```
/kaggle/working/PRAG/
└── src/
    ├── encode.py
    ├── utils.py
    └── data_aug/              ⬅️ 错误位置！
        └── 2wikimultihopqa/
```

**为什么错误？**
- `ROOT_DIR = '/kaggle/working/PRAG'`
- `DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'data_aug')` 
- 结果：`/kaggle/working/PRAG/data_aug`
- **不是** `/kaggle/working/PRAG/src/data_aug`

### ✅ 正确的结构（data_aug与src同级）

```
/kaggle/working/PRAG/
├── src/
│   ├── encode.py
│   ├── utils.py
│   └── root_dir_path.py
├── data_aug/                  ⬅️ 正确位置！与src同级
│   └── 2wikimultihopqa/
│       └── llama3.2-1b-instruct/
│           └── total.json
├── offline/                   # encode生成
└── output/                    # inference生成
```

## 快速检查清单

运行encode前，请确认：

- [ ] `augment_model=None` （Python的None对象，不是字符串）
- [ ] `data_aug/` 与 `src/` 在同一级别
- [ ] `ROOT_DIR` 指向项目根目录（如 `/kaggle/working/PRAG`）
- [ ] 数据路径为：`{ROOT_DIR}/data_aug/{dataset}/{model_name}/total.json`
- [ ] 工作目录可写（使用 `/kaggle/working/`，不是 `/kaggle/input/`）

## 诊断命令

如果仍有问题，运行此诊断脚本：

```python
import os
import sys

print("=== PRAG 环境诊断 ===\n")

# 1. 检查工作目录
WORK_DIR = '/kaggle/working/PRAG'
print(f"1. 工作目录: {WORK_DIR}")
print(f"   存在: {os.path.exists(WORK_DIR)}")
print(f"   可写: {os.access(WORK_DIR, os.W_OK)}")

# 2. 检查src
src_dir = f'{WORK_DIR}/src'
print(f"\n2. 源代码目录: {src_dir}")
print(f"   存在: {os.path.exists(src_dir)}")
if os.path.exists(src_dir):
    files = [f for f in os.listdir(src_dir) if f.endswith('.py')]
    print(f"   Python文件: {files[:5]}")

# 3. 检查data_aug位置
data_aug_locations = [
    f'{WORK_DIR}/data_aug',           # 正确位置
    f'{WORK_DIR}/src/data_aug',       # 错误位置
]
print(f"\n3. data_aug位置检查:")
for loc in data_aug_locations:
    exists = os.path.exists(loc)
    status = "✓ 正确" if exists and 'src/data_aug' not in loc else ("❌ 错误位置" if exists else "✗ 不存在")
    print(f"   {status}: {loc}")

# 4. 检查ROOT_DIR配置
print(f"\n4. ROOT_DIR配置:")
sys.path.insert(0, f'{WORK_DIR}/src')
try:
    from root_dir_path import ROOT_DIR
    from utils import DATA_ROOT_DIR
    print(f"   ROOT_DIR = {ROOT_DIR}")
    print(f"   DATA_ROOT_DIR = {DATA_ROOT_DIR}")
    print(f"   匹配: {DATA_ROOT_DIR == f'{WORK_DIR}/data_aug'}")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")

# 5. 检查数据文件
print(f"\n5. 数据文件检查:")
DATASET = '2wikimultihopqa'
MODEL_NAME = 'llama3.2-1b-instruct'
data_path = f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}/total.json'
print(f"   期望路径: {data_path}")
print(f"   存在: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    import json
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"   样本数: {len(data)}")
else:
    print(f"\n   查找total.json:")
    !find {WORK_DIR} -name "total.json" 2>/dev/null | head -5

# 6. 参数检查
print(f"\n6. 参数类型检查:")
test_params = {
    'None对象': None,
    '空字符串': '',
    '字符串None': 'None',
}
for name, value in test_params.items():
    is_none = value is None
    print(f"   {name}: is None = {is_none}, type = {type(value).__name__}")

print("\n=== 诊断完成 ===")
```

## 总结

**IsADirectoryError的两个根本原因**：

1. **augment_model参数错误**：使用了空字符串 `''` 或字符串 `'None'` 而不是 Python的 `None` 对象
2. **data_aug位置错误**：放在了 `src/` 目录内，而不是与 `src/` 同级

**正确做法**：
- 使用 `augment_model=None`（不带引号）
- 将 `data_aug/` 放在项目根目录，与 `src/` 同级
- 确保路径为：`{ROOT_DIR}/data_aug/{dataset}/{model_name}/total.json`
