# Kaggle 数据路径问题快速修复指南

## 问题：TypeError when running encode

### 错误场景

您的数据路径是：
```
/kaggle/input/20260109/PRAG-main/data_aug/2wikimultihopqa/
```

运行代码时出现 `TypeError`。

### 根本原因

代码期望的数据结构是：
```
data_aug/
└── {dataset}/              # 如：2wikimultihopqa
    └── {model_name}/       # 如：llama3.2-1b-instruct  ⬅️ 缺少这一层！
        └── total.json
```

但您的数据结构可能是：
```
data_aug/
└── 2wikimultihopqa/
    └── total.json          # 直接在这里，缺少model_name层级
```

### 解决方案

#### 方案1：重新组织数据结构（推荐）

```python
import os
import shutil

# 1. 设置路径
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'
WORK_DIR = '/kaggle/working/PRAG'

# 2. 创建工作目录
os.makedirs(WORK_DIR, exist_ok=True)

# 3. 复制源代码
!cp -r {INPUT_BASE}/src {WORK_DIR}/

# 4. 重新组织数据结构
# 检查当前数据结构
dataset = '2wikimultihopqa'
model_name = 'llama3.2-1b-instruct'

source_data_dir = f'{INPUT_BASE}/data_aug/{dataset}'
target_data_dir = f'{WORK_DIR}/data_aug/{dataset}/{model_name}'

# 检查源数据是否直接包含json文件
if os.path.exists(f'{source_data_dir}/total.json'):
    print(f"✓ 检测到数据在 {source_data_dir}/")
    print(f"  需要添加 {model_name}/ 层级")
    
    # 创建正确的目录结构
    os.makedirs(target_data_dir, exist_ok=True)
    
    # 复制或链接数据文件
    for file in os.listdir(source_data_dir):
        if file.endswith('.json'):
            src = os.path.join(source_data_dir, file)
            dst = os.path.join(target_data_dir, file)
            if not os.path.exists(dst):
                # 使用符号链接（节省空间）
                os.symlink(src, dst)
                print(f"  链接: {file}")
    
    print(f"✓ 数据结构已修正为: {target_data_dir}/")
    
elif os.path.exists(f'{source_data_dir}/{model_name}'):
    print(f"✓ 数据结构正确: {source_data_dir}/{model_name}/")
    # 直接链接整个目录
    !ln -sf {source_data_dir} {WORK_DIR}/data_aug/{dataset}
    
else:
    print(f"❌ 在 {source_data_dir}/ 未找到数据")
    print(f"可用文件:")
    !ls -la {source_data_dir}

# 5. 验证最终结构
print("\n=== 验证数据结构 ===")
final_path = f'{WORK_DIR}/data_aug/{dataset}/{model_name}'
if os.path.exists(final_path):
    files = os.listdir(final_path)
    print(f"✓ 路径正确: {final_path}/")
    print(f"  包含文件: {files}")
    
    # 检查total.json
    if 'total.json' in files:
        import json
        with open(f'{final_path}/total.json', 'r') as f:
            data = json.load(f)
        print(f"  total.json 包含 {len(data)} 个样本")
    else:
        print(f"  ⚠️ 未找到 total.json")
else:
    print(f"❌ 路径不存在: {final_path}")
```

#### 方案2：手动创建目录结构（简单但需复制）

```python
import os
import shutil

WORK_DIR = '/kaggle/working/PRAG'
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'

# 创建完整的目录结构
os.makedirs(f'{WORK_DIR}/src', exist_ok=True)
os.makedirs(f'{WORK_DIR}/data_aug/2wikimultihopqa/llama3.2-1b-instruct', exist_ok=True)

# 复制源代码
!cp -r {INPUT_BASE}/src/* {WORK_DIR}/src/

# 复制数据到正确的位置（添加model_name层级）
!cp -r {INPUT_BASE}/data_aug/2wikimultihopqa/*.json \
      {WORK_DIR}/data_aug/2wikimultihopqa/llama3.2-1b-instruct/

# 验证
!ls -la {WORK_DIR}/data_aug/2wikimultihopqa/llama3.2-1b-instruct/
```

#### 方案3：如果数据已经有model_name层级

如果您的数据实际上是在 `data_aug/2wikimultihopqa/llama3.2-1b-instruct/`，只是您报告的路径不完整：

```python
import os

WORK_DIR = '/kaggle/working/PRAG'
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'

# 创建工作目录
os.makedirs(WORK_DIR, exist_ok=True)

# 复制源代码
!cp -r {INPUT_BASE}/src {WORK_DIR}/

# 直接链接data_aug目录
!ln -sf {INPUT_BASE}/data_aug {WORK_DIR}/data_aug

# 验证结构
dataset = '2wikimultihopqa'
model = 'llama3.2-1b-instruct'
check_path = f'{WORK_DIR}/data_aug/{dataset}/{model}'

if os.path.exists(check_path):
    print(f"✓ 路径正确: {check_path}")
    !ls {check_path}
else:
    print(f"❌ 路径不存在: {check_path}")
    print("可用路径:")
    !ls -R {WORK_DIR}/data_aug/ | head -20
```

### 完整工作代码（修正后）

```python
# ===== Cell 1: 安装依赖 =====
!pip install -q transformers==4.44.2 peft==0.13.2

# ===== Cell 2: 准备环境 =====
import os
import sys
import shutil

WORK_DIR = '/kaggle/working/PRAG'
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'
DATASET = '2wikimultihopqa'
MODEL_NAME = 'llama3.2-1b-instruct'

# 创建工作目录
os.makedirs(WORK_DIR, exist_ok=True)
print(f"工作目录: {WORK_DIR}")

# 复制源代码
!cp -r {INPUT_BASE}/src {WORK_DIR}/
print("✓ 源代码已复制")

# ===== Cell 3: 修正数据结构 =====
# 检查并修正数据结构
source_dataset = f'{INPUT_BASE}/data_aug/{DATASET}'
target_model_dir = f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}'

# 创建目标目录
os.makedirs(target_model_dir, exist_ok=True)

# 检查源数据位置
if os.path.exists(f'{source_dataset}/total.json'):
    # 情况1: 数据直接在dataset目录下（缺少model层）
    print(f"检测到数据在: {source_dataset}/")
    print("添加model_name层级...")
    for file in os.listdir(source_dataset):
        if file.endswith('.json'):
            src = f'{source_dataset}/{file}'
            dst = f'{target_model_dir}/{file}'
            if not os.path.exists(dst):
                os.symlink(src, dst)
    print(f"✓ 已创建链接到: {target_model_dir}/")
    
elif os.path.exists(f'{source_dataset}/{MODEL_NAME}/total.json'):
    # 情况2: 数据结构已正确
    print(f"检测到正确的数据结构")
    target_dataset = f'{WORK_DIR}/data_aug/{DATASET}'
    os.makedirs(os.path.dirname(target_dataset), exist_ok=True)
    if not os.path.exists(target_dataset):
        os.symlink(source_dataset, target_dataset)
    print(f"✓ 已链接: {source_dataset}/ -> {target_dataset}/")
    
else:
    print(f"❌ 未找到数据文件!")
    print(f"请检查路径: {source_dataset}/")
    !ls -la {source_dataset}

# 验证最终结构
print("\n=== 验证数据结构 ===")
final_json = f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}/total.json'
if os.path.exists(final_json):
    import json
    with open(final_json, 'r') as f:
        data = json.load(f)
    print(f"✓ 找到数据文件: {final_json}")
    print(f"  样本数量: {len(data)}")
    print(f"  第一个样本的键: {list(data[0].keys())}")
else:
    print(f"❌ 未找到: {final_json}")
    print("\n当前data_aug结构:")
    !ls -R {WORK_DIR}/data_aug/ | head -30

# ===== Cell 4: 配置ROOT_DIR =====
sys.path.insert(0, f'{WORK_DIR}/src')

# 动态设置ROOT_DIR
import root_dir_path
root_dir_path.ROOT_DIR = WORK_DIR

# 或修改文件
with open(f'{WORK_DIR}/src/root_dir_path.py', 'w') as f:
    f.write(f'ROOT_DIR = "{WORK_DIR}"\n')

from root_dir_path import ROOT_DIR
print(f"✓ ROOT_DIR = {ROOT_DIR}")

# ===== Cell 5: 运行Encode =====
from encode import main as encode_main
import argparse

args_encode = argparse.Namespace(
    model_name=MODEL_NAME,
    dataset=DATASET,
    data_type=None,
    with_cot=True,
    sample=3,  # 先测试3个样本
    augment_model=None,  # 注意：None对象，不是字符串
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32
)

print("=== 开始Encode训练 ===")
print(f"参数: {args_encode}")

try:
    encode_main(args_encode)
    print("\n✓ Encode完成！")
    
    # 检查输出
    offline_dir = f'{WORK_DIR}/offline'
    if os.path.exists(offline_dir):
        print(f"\n生成的LoRA权重:")
        !find {offline_dir} -name "adapter_model.safetensors" | head -5
    else:
        print(f"⚠️ offline目录未创建")
        
except Exception as e:
    print(f"\n❌ Encode失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 诊断信息
    print("\n=== 诊断信息 ===")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"数据路径: {WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}/")
    print(f"数据存在: {os.path.exists(f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}/total.json')}")
```

### 快速诊断脚本

使用这个脚本快速诊断您的数据结构问题：

```python
import os
import json

WORK_DIR = '/kaggle/working/PRAG'
INPUT_BASE = '/kaggle/input/20260109/PRAG-main'
DATASET = '2wikimultihopqa'
MODEL_NAME = 'llama3.2-1b-instruct'

print("=== 数据结构诊断 ===\n")

# 检查各种可能的路径
paths_to_check = [
    (f'{INPUT_BASE}/data_aug/{DATASET}/total.json', 
     '数据直接在dataset下（缺少model层）'),
    (f'{INPUT_BASE}/data_aug/{DATASET}/{MODEL_NAME}/total.json', 
     '数据结构正确'),
    (f'{WORK_DIR}/data_aug/{DATASET}/{MODEL_NAME}/total.json', 
     '工作目录中的数据'),
]

for path, description in paths_to_check:
    exists = os.path.exists(path)
    print(f"{'✓' if exists else '✗'} {description}")
    print(f"  路径: {path}")
    if exists and path.endswith('.json'):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"  样本数: {len(data)}")
        except:
            print(f"  无法读取JSON")
    print()

# 显示实际的目录结构
print("=== 实际目录结构 ===")
data_aug_base = f'{INPUT_BASE}/data_aug'
if os.path.exists(data_aug_base):
    print(f"\n{data_aug_base}/")
    for item in os.listdir(data_aug_base):
        item_path = os.path.join(data_aug_base, item)
        if os.path.isdir(item_path):
            print(f"├── {item}/")
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    print(f"│   ├── {subitem}/")
                    files = [f for f in os.listdir(subitem_path) if f.endswith('.json')]
                    for f in files[:3]:
                        print(f"│   │   └── {f}")
                else:
                    print(f"│   └── {subitem}")
else:
    print(f"❌ 未找到: {data_aug_base}")

print("\n=== 建议 ===")
if os.path.exists(f'{INPUT_BASE}/data_aug/{DATASET}/total.json'):
    print("⚠️ 您的数据缺少model_name层级")
    print("请使用上面的'方案1'或'方案2'重新组织数据结构")
elif os.path.exists(f'{INPUT_BASE}/data_aug/{DATASET}/{MODEL_NAME}/total.json'):
    print("✓ 您的数据结构正确")
    print("确保在运行encode前正确链接或复制到工作目录")
else:
    print("❌ 未找到数据文件")
    print(f"请检查 {INPUT_BASE}/data_aug/ 目录")
```

### 总结

**TypeError的根本原因**：数据目录缺少 `{model_name}` 层级。

**解决步骤**：
1. 检查您的数据是在 `data_aug/2wikimultihopqa/` 还是 `data_aug/2wikimultihopqa/llama3.2-1b-instruct/`
2. 如果缺少model_name层，使用上述方案重新组织
3. 确保最终路径为：`{WORK_DIR}/data_aug/2wikimultihopqa/llama3.2-1b-instruct/total.json`
4. 运行诊断脚本确认结构正确
5. 运行encode代码
