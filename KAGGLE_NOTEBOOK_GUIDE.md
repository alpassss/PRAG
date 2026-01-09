# PRAG在Kaggle Notebook中的使用指南

## 一、代码的输入输出说明

### 1. 增强数据读取 (load_data函数)

**位置**: `src/utils.py` 第78-119行

**读取路径**:
```python
# 从 data_aug/{dataset}/{model_name}/ 读取
input_dir = os.path.join(DATA_ROOT_DIR, data_name, model_name)
# DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data_aug")
```

**示例**:
```
data_aug/
└── 2wikimultihopqa/
    └── llama3.2-1b-instruct/
        ├── total.json           # 被 load_data 读取
        ├── comparison.json      # 如果有多个类型
        └── bridge.json
```

### 2. Encode结果存储 (encode.py)

**位置**: `src/encode.py` 第160-179行

**存储路径**:
```python
output_dir = os.path.join(
    ROOT_DIR, 
    "offline", 
    args.model_name,                           # 如: llama3.2-1b-instruct
    f"rank={args.lora_rank}_alpha={args.lora_alpha}",  # 如: rank=2_alpha=32
    args.dataset,                              # 如: 2wikimultihopqa
    f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
    f"aug_model={args.augment_model}",
    filename,                                  # 如: total
)
```

**存储结构**:
```
offline/
└── llama3.2-1b-instruct/
    └── rank=2_alpha=32/
        ├── base_weight/                    # 基础LoRA权重
        │   └── adapter_model.safetensors
        └── 2wikimultihopqa/
            └── lr=0.0003_epoch=1_cot/
                └── aug_model=llama3.2-1b-instruct/
                    └── total/
                        ├── data_0/         # 第0个问题
                        │   ├── passage_0/  # 第0个文档的LoRA
                        │   │   ├── adapter_config.json
                        │   │   └── adapter_model.safetensors
                        │   ├── passage_1/
                        │   └── passage_2/
                        ├── data_1/         # 第1个问题
                        └── ...
```

### 3. Inference读取LoRA权重 (inference.py)

**位置**: `src/inference.py` 第23-31行, 78-88行

**读取路径**:
```python
load_adapter_path = os.path.join(
    ROOT_DIR, 
    "offline", 
    args.model_name, 
    f"rank={args.lora_rank}_alpha={args.lora_alpha}",
    args.dataset,
    f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
    f"aug_model={args.augment_model}",
)

# 对每个问题加载其对应的LoRA
adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
```

### 4. Inference结果保存 (inference.py)

**位置**: `src/inference.py` 第32-47行, 103-115行

**存储路径**:
```python
output_root_dir = os.path.join(
    ROOT_DIR, 
    "output",
    args.model_name, 
    f"rank={args.lora_rank}_alpha={args.lora_alpha}",
    args.dataset,
    f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
    f"aug_model={args.augment_model}",
    args.inference_method,  # icl / prag / combine
)
```

**存储文件**:
```
output/
└── llama3.2-1b-instruct/
    └── rank=2_alpha=32/
        └── 2wikimultihopqa/
            └── lr=0.0003_epoch=1_cot/
                └── aug_model=llama3.2-1b-instruct/
                    └── combine/
                        └── total/
                            ├── config.json      # 运行配置
                            ├── predict.json     # 每个问题的预测结果
                            └── result.txt       # 评估指标汇总
```

## 二、在Kaggle Notebook中的使用方法

### 方法1：直接调用Python脚本（推荐）

#### 步骤1：上传并解压数据

```python
# 在Kaggle Notebook中
import os
import tarfile

# 假设你已经上传了 data_aug.tar.gz 到 /kaggle/input/
# 解压到工作目录
!tar -xzvf /kaggle/input/prag-data/data_aug.tar.gz -C /kaggle/working/

# 克隆或上传PRAG代码
!git clone https://github.com/alpassss/PRAG.git /kaggle/working/PRAG
# 或者直接上传代码文件
```

#### 步骤2：配置ROOT_DIR

```python
# 修改 src/root_dir_path.py
with open('/kaggle/working/PRAG/src/root_dir_path.py', 'w') as f:
    f.write('ROOT_DIR = "/kaggle/working/PRAG"\n')
```

#### 步骤3：安装依赖

```python
!pip install -q transformers==4.44.2 peft==0.13.2 torch
```

#### 步骤4：运行Encode

```python
# 方式A: 使用 ! 命令
!cd /kaggle/working/PRAG && python src/encode.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --sample 10 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot

# 方式B: 在Python中执行
import sys
sys.path.insert(0, '/kaggle/working/PRAG/src')

import argparse
from encode import main as encode_main

args = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type=None,
    with_cot=True,
    sample=10,
    augment_model=None,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32
)

encode_main(args)
```

#### 步骤5：运行Inference

```python
# 方式A: 使用 ! 命令
!cd /kaggle/working/PRAG && python src/inference.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --sample 10 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --max_new_tokens 128 \
    --inference_method combine \
    --with_cot

# 方式B: 在Python中执行
from inference import main as inference_main

args = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type=None,
    with_cot=True,
    sample=10,
    augment_model=None,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    max_new_tokens=128,
    inference_method='combine'
)

inference_main(args)
```

#### 步骤6：查看结果

```python
import json

# 读取预测结果
with open('/kaggle/working/PRAG/output/llama3.2-1b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=llama3.2-1b-instruct/combine/total/predict.json', 'r') as f:
    predictions = json.load(f)

# 显示前几个结果
for pred in predictions[:3]:
    print(f"Question: {pred['question']}")
    print(f"Answer: {pred['answer']}")
    print(f"Predicted: {pred['eval_predict']}")
    print(f"EM: {pred['em']}, F1: {pred['f1']}")
    print("-" * 80)

# 读取评估指标
with open('/kaggle/working/PRAG/output/llama3.2-1b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=llama3.2-1b-instruct/combine/total/result.txt', 'r') as f:
    results = f.read()
    print(results)
```

### 方法2：封装为函数调用（更灵活）

创建一个新的notebook cell，封装功能：

```python
import os
import sys
import json
import argparse

# 添加路径
sys.path.insert(0, '/kaggle/working/PRAG/src')

# 设置ROOT_DIR
import root_dir_path
root_dir_path.ROOT_DIR = '/kaggle/working/PRAG'

def run_encode(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    sample=10,
    with_cot=True,
    lora_rank=2,
    lora_alpha=32,
    learning_rate=0.0003,
    num_train_epochs=1
):
    """运行文档参数化训练"""
    from encode import main as encode_main
    
    args = argparse.Namespace(
        model_name=model_name,
        dataset=dataset,
        data_type=None,
        with_cot=with_cot,
        sample=sample,
        augment_model=None,
        per_device_train_batch_size=1,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha
    )
    
    encode_main(args)
    print(f"✓ Encode完成！LoRA权重保存在:")
    print(f"  /kaggle/working/PRAG/offline/{model_name}/rank={lora_rank}_alpha={lora_alpha}/{dataset}/")

def run_inference(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    sample=10,
    with_cot=True,
    lora_rank=2,
    lora_alpha=32,
    learning_rate=0.0003,
    num_train_epochs=1,
    inference_method='combine',
    max_new_tokens=128
):
    """运行推理"""
    from inference import main as inference_main
    
    args = argparse.Namespace(
        model_name=model_name,
        dataset=dataset,
        data_type=None,
        with_cot=with_cot,
        sample=sample,
        augment_model=None,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_new_tokens=max_new_tokens,
        inference_method=inference_method
    )
    
    inference_main(args)
    
    # 返回结果路径
    output_dir = f"/kaggle/working/PRAG/output/{model_name}/rank={lora_rank}_alpha={lora_alpha}/{dataset}/lr={learning_rate}_epoch={num_train_epochs}_{'cot' if with_cot else 'direct'}/aug_model={model_name}/{inference_method}/total/"
    
    print(f"✓ Inference完成！结果保存在:")
    print(f"  {output_dir}")
    
    return output_dir

def load_results(output_dir):
    """加载并显示结果"""
    # 读取预测
    with open(os.path.join(output_dir, 'predict.json'), 'r') as f:
        predictions = json.load(f)
    
    # 读取评估
    with open(os.path.join(output_dir, 'result.txt'), 'r') as f:
        results = f.read()
    
    return predictions, results

# 使用示例
if __name__ == '__main__':
    # 1. 运行encode
    run_encode(
        model_name='llama3.2-1b-instruct',
        dataset='2wikimultihopqa',
        sample=5,  # 先测试5个样本
        with_cot=True
    )
    
    # 2. 运行inference
    output_dir = run_inference(
        model_name='llama3.2-1b-instruct',
        dataset='2wikimultihopqa',
        sample=5,
        with_cot=True,
        inference_method='prag'  # 或 'icl' 或 'combine'
    )
    
    # 3. 查看结果
    predictions, results = load_results(output_dir)
    
    print("\n=== 评估结果 ===")
    print(results)
    
    print("\n=== 前3个预测 ===")
    for pred in predictions[:3]:
        print(f"Q: {pred['question']}")
        print(f"A: {pred['answer']}")
        print(f"P: {pred['eval_predict']}")
        print(f"EM: {pred['em']}, F1: {pred['f1']}\n")
```

## 三、完整的Kaggle Notebook示例

```python
# Cell 1: 安装依赖
!pip install -q transformers==4.44.2 peft==0.13.2

# Cell 2: 准备数据和代码
import os
import shutil

# 创建工作目录
os.makedirs('/kaggle/working/PRAG', exist_ok=True)

# 解压增强数据（假设已上传为Kaggle Dataset）
!tar -xzf /kaggle/input/prag-augmented-data/data_aug.tar.gz -C /kaggle/working/PRAG/

# 克隆代码（或从Dataset上传）
!git clone https://github.com/alpassss/PRAG.git /tmp/prag_code
!cp -r /tmp/prag_code/src /kaggle/working/PRAG/
!cp -r /tmp/prag_code/configs /kaggle/working/PRAG/

# 配置ROOT_DIR
with open('/kaggle/working/PRAG/src/root_dir_path.py', 'w') as f:
    f.write('ROOT_DIR = "/kaggle/working/PRAG"\n')

# Cell 3: 导入和配置
import sys
sys.path.insert(0, '/kaggle/working/PRAG/src')

from encode import main as encode_main
from inference import main as inference_main
import argparse
import json

# Cell 4: 运行Encode（文档参数化）
args_encode = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type=None,
    with_cot=True,
    sample=10,  # 只处理10个样本用于测试
    augment_model=None,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32
)

print("开始Encode训练...")
encode_main(args_encode)
print("✓ Encode完成！")

# Cell 5: 运行Inference（推理）
args_inference = argparse.Namespace(
    model_name='llama3.2-1b-instruct',
    dataset='2wikimultihopqa',
    data_type=None,
    with_cot=True,
    sample=10,
    augment_model=None,
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    max_new_tokens=128,
    inference_method='combine'  # 使用参数+上下文
)

print("开始Inference推理...")
inference_main(args_inference)
print("✓ Inference完成！")

# Cell 6: 查看结果
result_dir = '/kaggle/working/PRAG/output/llama3.2-1b-instruct/rank=2_alpha=32/2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=llama3.2-1b-instruct/combine/total/'

# 读取结果
with open(os.path.join(result_dir, 'predict.json'), 'r') as f:
    predictions = json.load(f)

with open(os.path.join(result_dir, 'result.txt'), 'r') as f:
    results = f.read()

print("=== 评估指标 ===")
print(results)

print("\n=== 详细预测结果 ===")
import pandas as pd
df = pd.DataFrame(predictions)
print(df[['question', 'answer', 'eval_predict', 'em', 'f1']].head(10))
```

## 四、常见问题

### Q1: 内存不足怎么办？

**A**: Kaggle免费版GPU内存有限(16GB)，建议：
- 减少sample数量（从300改为10-50）
- 使用更小的模型（llama3.2-1b-instruct而不是llama3-8b-instruct）
- 使用Kaggle的P100或T4 GPU环境

### Q2: 如何只运行inference不运行encode？

**A**: 如果你已经有训练好的LoRA权重：
```python
# 1. 上传或下载预训练的offline文件夹到Kaggle
# 2. 确保路径正确
# 3. 直接运行inference

# 只要offline目录下有对应的LoRA权重，inference.py会自动加载
```

### Q3: 如何保存和下载结果？

**A**: Kaggle会自动保存output目录的内容：
```python
# 打包结果
!tar -czf /kaggle/working/prag_results.tar.gz -C /kaggle/working/PRAG/output .

# 在Kaggle UI中下载 prag_results.tar.gz
# 或者将结果复制到 /kaggle/working/ 目录（这个目录的内容可以下载）
```

### Q4: 数据路径找不到怎么办？

**A**: 检查以下几点：
1. `ROOT_DIR` 是否正确设置为 `/kaggle/working/PRAG`
2. `data_aug` 目录是否正确解压到 `/kaggle/working/PRAG/data_aug/`
3. 检查目录结构：
```python
!ls -la /kaggle/working/PRAG/data_aug/2wikimultihopqa/llama3.2-1b-instruct/
# 应该看到 total.json 等文件
```

### Q5: 如何验证encode是否成功？

**A**: 检查offline目录：
```python
import os

base_path = '/kaggle/working/PRAG/offline/llama3.2-1b-instruct/rank=2_alpha=32/'
print("Base weight:", os.path.exists(os.path.join(base_path, 'base_weight/adapter_model.safetensors')))

# 检查训练的LoRA
lora_path = os.path.join(base_path, '2wikimultihopqa/lr=0.0003_epoch=1_cot/aug_model=llama3.2-1b-instruct/total/data_0/passage_0/')
print("First LoRA:", os.path.exists(os.path.join(lora_path, 'adapter_model.safetensors')))

# 列出所有生成的LoRA
!find /kaggle/working/PRAG/offline -name "adapter_model.safetensors" | head -20
```

## 五、性能优化建议

### 1. 渐进式测试

```python
# 先用少量样本测试流程
run_encode(sample=3)
run_inference(sample=3)

# 确认流程正确后，再增加样本数
run_encode(sample=50)
run_inference(sample=50)
```

### 2. 使用Kaggle的持久化存储

```python
# 将中间结果保存到 /kaggle/working/
# 这样可以在不同的notebook session之间共享数据
```

### 3. 监控GPU使用

```python
# 在训练过程中监控GPU
import subprocess
subprocess.run(['nvidia-smi'])
```

## 六、总结

代码的I/O操作都是存在的，主要通过以下函数实现：

1. **读取增强数据**: `utils.py` 中的 `load_data()` 函数从 `data_aug/` 读取
2. **保存LoRA权重**: `encode.py` 自动保存到 `offline/` 目录
3. **加载LoRA权重**: `inference.py` 从 `offline/` 加载
4. **保存推理结果**: `inference.py` 保存到 `output/` 目录

在Kaggle Notebook中使用时，最重要的是：
- 正确设置 `ROOT_DIR`
- 确保数据目录结构正确
- 使用适当的参数（特别是sample数量）避免内存溢出

希望这个指南能帮助你在Kaggle上顺利运行PRAG！
