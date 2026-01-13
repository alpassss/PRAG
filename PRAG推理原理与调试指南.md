# PRAG推理原理详解与调试指南

## 一、PRAG推理的核心原理

### 1.1 三种模式的区别

#### ICL模式（In-Context Learning）
```
输入: 问题 + 3个检索到的passages
模型: 基础模型（无LoRA）
输出: 根据prompt中的passages回答
```

#### PRAG模式（Parametric RAG）
```
输入: 仅问题（无passages）
模型: 基础模型 + 合并的LoRA适配器
输出: 根据LoRA中参数化的知识回答
```

#### COMBINE模式
```
输入: 问题 + 3个passages
模型: 基础模型 + 合并的LoRA适配器
输出: 同时利用LoRA知识和prompt中的passages
```

### 1.2 LoRA权重的加载过程

对于每个测试样本（例如test_id=0），inference会：

1. **确定需要加载哪些LoRA权重**
   - 根据该样本有多少个passages（通常是3个）
   - 每个passage对应一个训练好的LoRA适配器

2. **构建LoRA权重路径**
   ```python
   # 基础路径
   base_path = "offline/qwen2.5-1.5b-instruct/rank=2_alpha=32/popqa/lr=0.0003_epoch=1_direct/aug_model=qwen2.5-1.5b-instruct/"
   
   # 对于test_id=0的样本，需要加载：
   adapter_0 = base_path + "total/data_0/passage_0/adapter_model.safetensors"
   adapter_1 = base_path + "total/data_0/passage_1/adapter_model.safetensors"
   adapter_2 = base_path + "total/data_0/passage_2/adapter_model.safetensors"
   ```

3. **依次加载适配器**
   ```python
   # 第一个适配器（不指定名称，使用默认）
   model = PeftModel.from_pretrained(model, adapter_0)
   # 自动分配名称，通常是 "default"
   
   # 第二个适配器（指定名称）
   model.load_adapter(adapter_1, adapter_name="adapter_1")
   
   # 第三个适配器（指定名称）
   model.load_adapter(adapter_2, adapter_name="adapter_2")
   ```

4. **合并所有适配器**
   ```python
   model.add_weighted_adapter(
       adapters = ["default", "adapter_1", "adapter_2"],
       weights = [1/3, 1/3, 1/3],  # 平均权重
       adapter_name = "merge",
       combination_type = "linear"  # 线性组合
   )
   ```

5. **激活合并后的适配器**
   ```python
   model.set_adapter("merge")
   # 现在模型使用的是合并后的适配器
   ```

## 二、路径匹配的关键

### 2.1 encode和inference参数必须完全一致

LoRA权重的保存路径由这些参数决定：
- `model_name`: qwen2.5-1.5b-instruct
- `lora_rank`: 2
- `lora_alpha`: 32
- `dataset`: popqa
- `learning_rate`: 0.0003
- `num_train_epochs`: 1 或 2
- `with_cot`: False (直接回答) 或 True (思维链)
- `augment_model`: qwen2.5-1.5b-instruct

**如果任何一个参数不匹配，inference会尝试加载不存在的路径！**

### 2.2 路径结构示例

```
offline/
└── qwen2.5-1.5b-instruct/
    └── rank=2_alpha=32/
        └── popqa/
            └── lr=0.0003_epoch=1_direct/
                └── aug_model=qwen2.5-1.5b-instruct/
                    └── total/
                        ├── data_0/
                        │   ├── passage_0/
                        │   │   └── adapter_model.safetensors
                        │   ├── passage_1/
                        │   │   └── adapter_model.safetensors
                        │   └── passage_2/
                        │       └── adapter_model.safetensors
                        ├── data_1/
                        │   ├── passage_0/
                        │   ├── passage_1/
                        │   └── passage_2/
                        ...
                        └── data_299/
                            ├── passage_0/
                            ├── passage_1/
                            └── passage_2/
```

## 三、使用调试模式

### 3.1 运行调试版本

我创建了一个带详细调试信息的inference_debug.py：

```python
# 添加 --debug 参数
args_inference = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='popqa',
    data_type='total',
    with_cot=False,
    augment_model='qwen2.5-1.5b-instruct',
    num_train_epochs=1,  # 必须与encode一致！
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    inference_method='combine',
    sample=10,  # 小样本测试
    max_new_tokens=20,
    debug=True  # ✅ 开启调试模式
)

# 使用调试版本
from inference_debug import main as inference_main
inference_main(args_inference)
```

### 3.2 调试输出会显示

对于前3个样本，会打印：

1. **基础路径信息**
   ```
   基础路径: offline/.../
   路径存在: True/False
   路径内容: [列出所有文件]
   ```

2. **每个适配器的详细信息**
   ```
   --- Passage 0 ---
   适配器路径: offline/.../data_0/passage_0/
   路径存在: True/False
   路径内文件: ['adapter_model.safetensors', 'adapter_config.json']
   adapter_model.safetensors 大小: 45,678 bytes (44.61 KB)
   ✓ 成功加载第一个适配器
   分配的适配器名称: default
   ```

3. **合并信息**
   ```
   已加载的所有适配器: ['default', 'adapter_1', 'adapter_2']
   将要合并的适配器: ['default', 'adapter_1', 'adapter_2']
   ✓ 成功合并适配器为 'merge'
   权重: [0.333, 0.333, 0.333]
   合并方式: linear
   当前激活的适配器: merge
   ```

4. **参数检查**
   ```
   模型参数统计:
   总参数数量: 1234
   LoRA参数数量: 120
   LoRA参数非零: True/False
   ```

## 四、常见问题诊断

### 问题1: 路径不存在

**症状**：调试输出显示 `路径存在: False`

**原因**：encode和inference参数不匹配

**解决**：
```python
# 检查encode时使用的参数
print("Encode参数:")
print(f"  num_train_epochs: {encode_epochs}")
print(f"  learning_rate: {encode_lr}")

# 确保inference使用相同参数
print("Inference参数:")
print(f"  num_train_epochs: {inference_epochs}")  # 必须相同！
print(f"  learning_rate: {inference_lr}")  # 必须相同！
```

### 问题2: 文件大小为0或很小

**症状**：`adapter_model.safetensors 大小: 0 bytes`

**原因**：训练失败或没有正确保存

**解决**：重新运行encode，检查训练过程是否有错误

### 问题3: LoRA参数非零为False

**症状**：`LoRA参数非零: False`

**原因**：适配器加载了但都是零值（未训练或训练失败）

**解决**：
1. 检查encode的训练损失是否下降
2. 重新训练（可能需要更多epochs）

### 问题4: 适配器加载成功但结果不变

**症状**：所有调试信息都显示正常，但F1分数不变

**可能原因**：
1. **正在查看旧结果**（最常见）
   ```python
   import shutil
   shutil.rmtree('output', ignore_errors=True)
   # 然后重新运行
   ```

2. **训练的数据质量问题**
   - LoRA可能确实学到了知识，但不够充分
   - 尝试增加训练轮数或调整学习率

3. **模型本身的限制**
   - 1.5B参数的模型能力有限
   - 论文中的结果可能使用了更好的训练数据

## 五、完整的调试流程

### 步骤1：运行调试版本
```python
# 使用 inference_debug.py 并设置 debug=True
# 只测试少量样本（sample=3）以减少输出
```

### 步骤2：检查基础路径
```
✓ 基础路径存在？
✓ 包含 "total" 目录？
```

### 步骤3：检查每个适配器
```
✓ data_0/passage_0/ 存在？
✓ adapter_model.safetensors 存在？
✓ 文件大小 > 0？（通常几十KB）
✓ 成功加载？
```

### 步骤4：检查合并
```
✓ 所有适配器都加载了？
✓ 合并成功？
✓ "merge"被激活？
```

### 步骤5：检查参数
```
✓ LoRA参数数量 > 0？
✓ LoRA参数非零？
```

### 步骤6：比较结果
```python
# 确保删除旧结果
shutil.rmtree('output', ignore_errors=True)

# 运行ICL
args.inference_method = 'icl'
inference_main(args)

# 运行COMBINE  
args.inference_method = 'combine'
inference_main(args)

# 比较F1分数
# COMBINE应该 >= ICL
```

## 六、预期行为

### 正常情况下
```
DEBUG: 样本 0 的适配器加载详情
问题: What is George Rankin's occupation?
Passages数量: 3

--- Passage 0 ---
适配器路径: offline/.../data_0/passage_0/
路径存在: True
路径内文件: ['adapter_model.safetensors', 'adapter_config.json', ...]
adapter_model.safetensors 大小: 45,678 bytes (44.61 KB)
✓ 成功加载第一个适配器
  分配的适配器名称: default

--- Passage 1 ---
适配器路径: offline/.../data_0/passage_1/
路径存在: True
adapter_model.safetensors 大小: 45,680 bytes (44.61 KB)
✓ 成功加载适配器 'adapter_1'

--- Passage 2 ---
适配器路径: offline/.../data_0/passage_2/
路径存在: True
adapter_model.safetensors 大小: 45,682 bytes (44.62 KB)
✓ 成功加载适配器 'adapter_2'

已加载的所有适配器: ['default', 'adapter_1', 'adapter_2']
✓ 成功合并适配器为 'merge'
当前激活的适配器: merge
LoRA参数非零: True
```

### 异常情况
如果看到任何 ❌ 标记，就找到了问题所在！

## 七、下一步

1. 运行 `inference_debug.py` 并设置 `debug=True`
2. 查看前3个样本的详细输出
3. 找到第一个 ❌ 标记
4. 根据上面的诊断指南解决问题
5. 如果所有 ✓ 但结果仍不变，删除output目录重新测试

如果所有检查都通过但仍有问题，请提供调试输出的完整内容！
