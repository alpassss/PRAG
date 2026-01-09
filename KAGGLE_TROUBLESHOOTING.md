# Kaggle Notebook 常见错误及解决方案

## 问题1: NotADirectoryError - ROOT_DIR配置错误

### 错误现象
```python
NotADirectoryError: [Errno 20] Not a directory
```

### 原因分析

您的配置有以下问题：

1. **ROOT_DIR设置错误** - 指向了文件而不是目录：
```python
# ❌ 错误 - 指向了文件本身
ROOT_DIR = "/kaggle/input/20260109/PRAG-main/src/root_dir_path.py"

# ✅ 正确 - 应该指向PRAG项目根目录
ROOT_DIR = "/kaggle/input/20260109/PRAG-main"
```

2. **数据路径嵌套错误** - `data_aug/data_aug` 多了一层：
```
# ❌ 错误的结构
/kaggle/input/20260109/PRAG-main/data_aug/data_aug/2wikimultihopqa/...

# ✅ 正确的结构
/kaggle/input/20260109/PRAG-main/data_aug/2wikimultihopqa/...
```

3. **augment_model参数错误** - 字符串'None'而不是Python的None：
```python
# ❌ 错误
augment_model='None'

# ✅ 正确
augment_model=None
```

4. **Kaggle只读问题** - `/kaggle/input/` 是只读的，无法写入结果：
   - Encode需要保存LoRA权重到 `offline/` 目录
   - Inference需要保存结果到 `output/` 目录
   - 必须使用可写的 `/kaggle/working/` 目录

### 完整解决方案

#### 步骤1: 正确准备目录结构

```python
import os
import shutil

# 创建工作目录
os.makedirs('/kaggle/working/PRAG', exist_ok=True)

# 方案A: 如果数据在Kaggle Dataset中
# 复制源代码到工作目录（可写）
!cp -r /kaggle/input/20260109/PRAG-main/src /kaggle/working/PRAG/
!cp -r /kaggle/input/20260109/PRAG-main/configs /kaggle/working/PRAG/

# 修正数据目录结构（如果是 data_aug/data_aug 的情况）
if os.path.exists('/kaggle/input/20260109/PRAG-main/data_aug/data_aug'):
    # 创建符号链接或复制
    !ln -s /kaggle/input/20260109/PRAG-main/data_aug/data_aug /kaggle/working/PRAG/data_aug
else:
    # 正常情况
    !ln -s /kaggle/input/20260109/PRAG-main/data_aug /kaggle/working/PRAG/data_aug

# 方案B: 如果需要完全复制（数据较小时）
# !cp -r /kaggle/input/20260109/PRAG-main/* /kaggle/working/PRAG/

# 验证目录结构
print("=== 验证目录结构 ===")
print("工作目录内容:")
!ls -la /kaggle/working/PRAG/

print("\n数据目录内容:")
!ls -la /kaggle/working/PRAG/data_aug/

print("\n具体数据集:")
!ls -la /kaggle/working/PRAG/data_aug/2wikimultihopqa/ 2>/dev/null || echo "未找到2wikimultihopqa"
```

#### 步骤2: 正确配置ROOT_DIR

```python
# 直接在代码中设置（推荐）
import sys
sys.path.insert(0, '/kaggle/working/PRAG/src')

# 动态修改ROOT_DIR
import root_dir_path
root_dir_path.ROOT_DIR = '/kaggle/working/PRAG'

# 或者修改文件
with open('/kaggle/working/PRAG/src/root_dir_path.py', 'w') as f:
    f.write('ROOT_DIR = "/kaggle/working/PRAG"\n')

# 验证配置
from root_dir_path import ROOT_DIR
print(f"ROOT_DIR = {ROOT_DIR}")
print(f"ROOT_DIR exists: {os.path.exists(ROOT_DIR)}")
print(f"ROOT_DIR is directory: {os.path.isdir(ROOT_DIR)}")
```

#### 步骤3: 验证数据路径

```python
import os

ROOT_DIR = '/kaggle/working/PRAG'

# 检查数据增强文件是否存在
data_aug_dir = os.path.join(ROOT_DIR, 'data_aug', '2wikimultihopqa', 'llama3.2-1b-instruct')
total_json = os.path.join(data_aug_dir, 'total.json')

print(f"数据目录: {data_aug_dir}")
print(f"目录存在: {os.path.exists(data_aug_dir)}")
print(f"total.json存在: {os.path.exists(total_json)}")

if os.path.exists(data_aug_dir):
    print(f"文件列表: {os.listdir(data_aug_dir)}")
else:
    print("❌ 数据目录不存在，请检查路径!")
```

#### 步骤4: 正确运行encode

```python
import argparse
from encode import main as encode_main

args_encode = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type=None,  # None，不是字符串 'None'
    with_cot=True,
    sample=10,
    augment_model=None,  # ✅ 这里是 None，不是字符串 'None'
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32
)

print("开始Encode训练...")
try:
    encode_main(args_encode)
    print("✓ Encode完成！")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
```

## 问题2: 数据路径不匹配

### 如果您的数据结构是 `data_aug/data_aug/`

这种情况通常是解压tar.gz时创建了额外的目录层级。

**解决方案1: 重新组织目录**
```python
import shutil
import os

# 如果是 data_aug/data_aug/ 结构
source = '/kaggle/input/20260109/PRAG-main/data_aug/data_aug'
target = '/kaggle/working/PRAG/data_aug'

if os.path.exists(source):
    # 复制或创建符号链接
    if os.path.exists(target):
        shutil.rmtree(target)
    
    # 选项A: 符号链接（节省空间）
    os.symlink(source, target)
    
    # 选项B: 复制（如果需要修改）
    # shutil.copytree(source, target)
    
    print(f"✓ 已修正数据路径")
    print(f"从: {source}")
    print(f"到: {target}")
```

**解决方案2: 修改utils.py中的DATA_ROOT_DIR**
```python
# 临时修改（不推荐，仅用于调试）
import sys
sys.path.insert(0, '/kaggle/working/PRAG/src')

import utils
# 如果数据在 data_aug/data_aug/ 下
utils.DATA_ROOT_DIR = '/kaggle/input/20260109/PRAG-main/data_aug/data_aug'
```

## 问题3: 完整的Kaggle Notebook模板

```python
# ===== Cell 1: 环境准备 =====
!pip install -q transformers==4.44.2 peft==0.13.2

# ===== Cell 2: 目录设置 =====
import os
import sys
import shutil

# 创建工作目录
WORK_DIR = '/kaggle/working/PRAG'
os.makedirs(WORK_DIR, exist_ok=True)

# 假设您的数据在 /kaggle/input/20260109/PRAG-main/
INPUT_DIR = '/kaggle/input/20260109/PRAG-main'

# 复制源代码（必须，因为需要可写目录）
print("复制源代码...")
if os.path.exists(f'{INPUT_DIR}/src'):
    !cp -r {INPUT_DIR}/src {WORK_DIR}/
else:
    print(f"❌ 找不到源代码目录: {INPUT_DIR}/src")

# 处理数据目录
print("\n处理数据目录...")
# 检查数据在哪里
if os.path.exists(f'{INPUT_DIR}/data_aug/data_aug'):
    # 情况1: 嵌套的 data_aug/data_aug
    print("检测到嵌套的data_aug目录，创建符号链接...")
    !ln -sf {INPUT_DIR}/data_aug/data_aug {WORK_DIR}/data_aug
elif os.path.exists(f'{INPUT_DIR}/data_aug'):
    # 情况2: 正常的 data_aug
    print("检测到正常的data_aug目录，创建符号链接...")
    !ln -sf {INPUT_DIR}/data_aug {WORK_DIR}/data_aug
else:
    print("❌ 找不到data_aug目录!")

# 验证
print("\n=== 验证目录结构 ===")
print(f"工作目录: {WORK_DIR}")
print(f"源代码存在: {os.path.exists(f'{WORK_DIR}/src')}")
print(f"数据目录存在: {os.path.exists(f'{WORK_DIR}/data_aug')}")

if os.path.exists(f'{WORK_DIR}/data_aug'):
    print(f"数据集列表:")
    !ls {WORK_DIR}/data_aug/

# ===== Cell 3: 配置ROOT_DIR =====
# 方法1: 修改文件
with open(f'{WORK_DIR}/src/root_dir_path.py', 'w') as f:
    f.write(f'ROOT_DIR = "{WORK_DIR}"\n')

# 方法2: 动态设置
sys.path.insert(0, f'{WORK_DIR}/src')
import root_dir_path
root_dir_path.ROOT_DIR = WORK_DIR

from root_dir_path import ROOT_DIR
print(f"✓ ROOT_DIR已设置为: {ROOT_DIR}")

# ===== Cell 4: 验证数据 =====
import os
import json

# 检查具体数据集
dataset = '2wikimultihopqa'
model = 'llama3.2-1b-instruct'

data_path = os.path.join(ROOT_DIR, 'data_aug', dataset, model)
total_json = os.path.join(data_path, 'total.json')

print(f"数据路径: {data_path}")
print(f"路径存在: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    files = os.listdir(data_path)
    print(f"文件列表: {files}")
    
    if 'total.json' in files:
        with open(total_json, 'r') as f:
            data = json.load(f)
        print(f"✓ total.json包含 {len(data)} 个样本")
        print(f"第一个样本: {list(data[0].keys())}")
    else:
        print("❌ 找不到total.json!")
else:
    print("❌ 数据路径不存在!")
    print("\n可用的数据集:")
    !ls {ROOT_DIR}/data_aug/

# ===== Cell 5: 运行Encode =====
import argparse
from encode import main as encode_main

args_encode = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type=None,  # ✅ Python的None对象
    with_cot=True,
    sample=3,  # 先测试3个样本
    augment_model=None,  # ✅ Python的None对象，不是字符串
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
    offline_dir = os.path.join(ROOT_DIR, 'offline')
    print(f"\n生成的LoRA权重:")
    !find {offline_dir} -name "adapter_model.safetensors" | head -5
    
except Exception as e:
    print(f"\n❌ Encode失败: {e}")
    import traceback
    traceback.print_exc()

# ===== Cell 6: 运行Inference =====
from inference import main as inference_main

args_inference = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type=None,
    with_cot=True,
    sample=3,
    augment_model=None,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    max_new_tokens=128,
    inference_method='prag'  # 或 'icl' 或 'combine'
)

print("=== 开始Inference推理 ===")
try:
    inference_main(args_inference)
    print("\n✓ Inference完成！")
    
    # 查看结果
    output_dir = os.path.join(ROOT_DIR, 'output')
    result_file = !find {output_dir} -name "result.txt" | head -1
    if result_file:
        print(f"\n结果文件: {result_file[0]}")
        !cat {result_file[0]}
    
except Exception as e:
    print(f"\n❌ Inference失败: {e}")
    import traceback
    traceback.print_exc()

# ===== Cell 7: 查看详细结果 =====
import json
import pandas as pd

# 找到预测文件
output_base = os.path.join(ROOT_DIR, 'output')
predict_files = !find {output_base} -name "predict.json"

if predict_files:
    predict_file = predict_files[0]
    print(f"预测文件: {predict_file}")
    
    with open(predict_file, 'r') as f:
        predictions = json.load(f)
    
    # 显示为表格
    df = pd.DataFrame(predictions)
    display(df[['question', 'answer', 'eval_predict', 'em', 'f1']])
    
    # 计算平均指标
    print(f"\n平均EM: {df['em'].astype(float).mean():.4f}")
    print(f"平均F1: {df['f1'].astype(float).mean():.4f}")
else:
    print("未找到预测结果文件")
```

## 检查清单

在运行代码之前，请确认：

- [ ] ROOT_DIR 指向目录，不是文件：`/kaggle/working/PRAG` ✅
- [ ] 数据路径正确：`/kaggle/working/PRAG/data_aug/2wikimultihopqa/llama3.2-1b-instruct/total.json` 存在
- [ ] augment_model 是 `None`，不是字符串 `'None'` ✅
- [ ] 工作目录可写：`/kaggle/working/` 而不是 `/kaggle/input/` ✅
- [ ] 源代码已复制到可写目录 ✅
- [ ] 环境变量和路径正确设置 ✅

## 快速诊断脚本

```python
import os
import sys

print("=== PRAG Kaggle 环境诊断 ===\n")

# 1. 检查ROOT_DIR
sys.path.insert(0, '/kaggle/working/PRAG/src')
try:
    from root_dir_path import ROOT_DIR
    print(f"✓ ROOT_DIR = {ROOT_DIR}")
    print(f"  - 存在: {os.path.exists(ROOT_DIR)}")
    print(f"  - 是目录: {os.path.isdir(ROOT_DIR)}")
    print(f"  - 可写: {os.access(ROOT_DIR, os.W_OK)}")
except Exception as e:
    print(f"❌ ROOT_DIR错误: {e}")

# 2. 检查数据
print("\n=== 数据检查 ===")
data_dir = os.path.join(ROOT_DIR, 'data_aug')
if os.path.exists(data_dir):
    datasets = os.listdir(data_dir)
    print(f"✓ 找到数据集: {datasets}")
    
    # 检查具体数据集
    for ds in datasets:
        ds_path = os.path.join(data_dir, ds)
        if os.path.isdir(ds_path):
            models = os.listdir(ds_path)
            print(f"  - {ds}: {models}")
            
            for model in models:
                model_path = os.path.join(ds_path, model)
                if os.path.isdir(model_path):
                    files = os.listdir(model_path)
                    print(f"    - {model}: {files}")
else:
    print(f"❌ 数据目录不存在: {data_dir}")

# 3. 检查源代码
print("\n=== 源代码检查 ===")
src_files = ['augment.py', 'encode.py', 'inference.py', 'utils.py']
for f in src_files:
    path = os.path.join(ROOT_DIR, 'src', f)
    exists = os.path.exists(path)
    print(f"{'✓' if exists else '❌'} {f}: {exists}")

# 4. 检查可写性
print("\n=== 可写性检查 ===")
test_dirs = ['offline', 'output']
for d in test_dirs:
    path = os.path.join(ROOT_DIR, d)
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, '.test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"✓ {d}/ 可写")
    except Exception as e:
        print(f"❌ {d}/ 不可写: {e}")

print("\n=== 诊断完成 ===")
```

运行这个诊断脚本可以快速发现问题所在！
