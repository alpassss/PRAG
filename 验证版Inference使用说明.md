# 验证版Inference使用说明

## 概述

`inference_with_verification.py` 是一个增强版的推理脚本，在推理过程中提供详细的验证信息，帮助您确认：

1. **LoRA适配器加载情况** - 哪些适配器被加载了
2. **适配器合并过程** - 多个适配器如何合并成一个
3. **模型状态验证** - 合并后的模型是否与原始模型不同
4. **评估过程验证** - 指标计算是否正确

## 主要功能

### 1. 详细的适配器加载日志

对于每个样本（前3个会显示详细信息），会显示：

```
[STEP 2] Loading LoRA adapters for 3 passages...

  Passage 0:
    Path: /path/to/offline/.../data_0/passage_0
    Exists: True
    Files: ['adapter_config.json', 'README.md', 'adapter_model.safetensors']
    adapter_model.safetensors size: 7,077,656 bytes (6911.77 KB)
    ✓ Loaded as adapter: 'default'

  Passage 1:
    Path: /path/to/offline/.../data_0/passage_1
    Exists: True
    Files: ['adapter_config.json', 'README.md', 'adapter_model.safetensors']
    adapter_model.safetensors size: 7,077,656 bytes (6911.77 KB)
    ✓ Loaded as adapter: 'adapter_1'
```

**解释**:
- 显示每个passage对应的适配器路径
- 验证文件是否存在
- 显示文件大小（相同大小是正常的，因为架构相同）
- 显示分配的适配器名称

### 2. 适配器合并验证

```
[STEP 3] All loaded adapters:
    0. 'default'
    1. 'adapter_1'
    2. 'adapter_2'

[STEP 4] Merging 3 adapters into single 'merge' adapter...
  Merge method: linear
  Weights: [0.333, 0.333, 0.333] (equal contribution)
  ✓ Merge successful - created adapter 'merge'

[STEP 5] Verifying merged adapter is active...
  ✓ Active adapter: 'merge' (correct)
```

**解释**:
- 列出所有已加载的适配器
- 显示合并方法（linear）和权重分配
- 确认合并成功
- 验证合并后的适配器已激活

### 3. 模型参数变化验证

```
[STEP 6] Comparing merged model with base model...
  ✓ Model parameters CHANGED by LoRA
    - 28/30 sampled parameters are different
    - Maximum difference: 0.523416
  ✅ Conclusion: Using LoRA-modified model
```

**解释**:
- 比较基础模型和LoRA修改后的模型
- 显示有多少参数发生了变化
- 显示最大参数差异
- 明确结论：是否在使用LoRA修改的模型

**如果看到这个**:
```
  ❌ Model parameters UNCHANGED
    - 0/30 sampled parameters are different
  ⚠ Warning: LoRA may not be active!
```
说明LoRA适配器没有生效，需要检查加载过程。

### 4. 预测生成和评估

```
[STEP 7] Generating prediction...
  Mode: PRAG (no passages in prompt, knowledge in LoRA)
  Generated answer: Muhammad Zia-ul-Haq
  Ground truth: Muhammad Zia-ul-Haq
  F1 Score: 1.0000
  EM Score: 1.0000
```

**解释**:
- 显示使用的模式（PRAG或COMBINE）
- 显示生成的答案和真实答案
- 显示该样本的F1和EM分数

### 5. 最终评估验证

```
[METRIC VERIFICATION]
  Total samples: 300
  Sample metric check (first 3 samples):
    Sample 0: F1=1.0, EM=1.0, Pred='Muhammad Zia-ul-Haq...', GT='Muhammad Zia-ul-Haq...'
    Sample 1: F1=0.8, EM=0.0, Pred='Pakistan...', GT='Islamabad...'
    Sample 2: F1=0.5, EM=0.0, Pred='India...', GT='New Delhi...'
```

**解释**:
- 验证总样本数
- 显示前几个样本的详细指标
- 可以手动检查计算是否正确

## 使用方法

### 在Kaggle Notebook中使用

```python
import sys
sys.path.insert(0, '/kaggle/working/PRAG/src')

from inference_with_verification import main as inference_main
import argparse

args_inference = argparse.Namespace(
    model_name='qwen2.5-1.5b-instruct',
    dataset='popqa',
    data_type='total',
    with_cot=False,
    augment_model='qwen2.5-1.5b-instruct',
    num_train_epochs=1,
    learning_rate=0.0003,
    lora_rank=2,
    lora_alpha=32,
    inference_method='combine',  # 'icl', 'prag', 或 'combine'
    sample=10,  # 先测试10个样本，看详细输出
    max_new_tokens=20,
)

inference_main(args_inference)
```

### 命令行使用

```bash
python src/inference_with_verification.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset popqa \
    --data_type total \
    --augment_model qwen2.5-1.5b-instruct \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --inference_method combine \
    --sample 10 \
    --max_new_tokens 20
```

## 三种推理模式对比

### ICL模式
- **使用**: 基础模型 + passages在prompt中
- **LoRA**: 不加载
- **输出**: 显示使用基础模型

### PRAG模式
- **使用**: LoRA修改的模型
- **Passages**: 不放在prompt中（知识在LoRA里）
- **输出**: 显示加载、合并、验证全过程

### COMBINE模式
- **使用**: LoRA修改的模型 + passages在prompt中
- **Passages**: 同时利用LoRA知识和prompt中的passages
- **输出**: 显示加载、合并、验证全过程

## 关键验证点

### ✓ 成功的验证输出应该显示:

1. **适配器加载**: 所有passage的适配器都成功加载
2. **适配器合并**: merge操作成功
3. **激活验证**: active_adapter = 'merge'
4. **参数变化**: 有参数发生了变化（changed > 0）
5. **指标计算**: F1/EM分数在合理范围内

### ❌ 如果看到问题:

1. **"Exists: False"**: 适配器文件不存在
   - 检查encode是否成功运行
   - 检查参数是否匹配（num_train_epochs等）

2. **"❌ Merge failed"**: 合并失败
   - 检查PEFT版本
   - 查看错误信息

3. **"Model parameters UNCHANGED"**: LoRA未生效
   - 适配器可能没有正确加载
   - 可能所有适配器都是base_weight的复制（训练失败）

4. **F1分数异常低**: 
   - 检查训练是否充分（num_train_epochs是否足够）
   - 使用check_lora_weights.py验证训练

## 输出文件

与原始inference.py相同：

- `output/.../predict.json` - 所有预测结果
- `output/.../result.txt` - 最终评估指标
- `output/.../config.json` - 运行配置

## 调试建议

1. **首次运行**: 使用 `sample=10` 只测试10个样本，查看详细输出
2. **验证成功后**: 去掉sample限制，运行完整数据集
3. **对比模式**: 依次运行ICL、PRAG、COMBINE，比较结果
4. **预期结果**: COMBINE >= PRAG > ICL（如果训练充分）

## 与其他工具配合使用

1. **先用verify_lora_active.py**: 验证单个适配器是否训练成功
2. **再用inference_with_verification.py**: 验证整个推理流程
3. **最后用check_lora_weights.py**: 如果结果仍不理想，检查权重是否变化

## 示例输出解读

### 成功案例

```
[STEP 6] Comparing merged model with base model...
  ✓ Model parameters CHANGED by LoRA
    - 28/30 sampled parameters are different
    - Maximum difference: 0.523416
  ✅ Conclusion: Using LoRA-modified model

Results on 10 samples:
  EM: 0.2000
  F1: 0.3500
  PREC: 0.4000
  RECALL: 0.3800
```

✓ 参数变化了，说明LoRA在工作
✓ F1=0.35 > ICL基线，说明LoRA有帮助

### 问题案例

```
[STEP 6] Comparing merged model with base model...
  ❌ Model parameters UNCHANGED
    - 0/30 sampled parameters are different
  ⚠ Warning: LoRA may not be active!

Results on 10 samples:
  EM: 0.1000
  F1: 0.1000
  PREC: 0.1200
  RECALL: 0.1100
```

❌ 参数没变化，LoRA没生效
❌ F1=0.10 = ICL基线，确认LoRA无效
→ 需要检查训练或加载过程

## 总结

这个验证版inference脚本提供了完整的可观测性，让您可以：

1. ✓ 确认每个适配器是否加载
2. ✓ 确认适配器合并是否成功
3. ✓ 确认使用的是LoRA修改的模型而非基础模型
4. ✓ 确认指标计算是否正确
5. ✓ 识别问题出在哪个环节

如果所有验证都通过但F1分数仍然低，那问题就在训练不充分（num_train_epochs太少），而不是代码bug。
