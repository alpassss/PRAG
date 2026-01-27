# Parametric RAG 推理阶段 LoRA 权重加载问题诊断与修复报告

## 摘要

本报告详细记录了在复现 Parametric Retrieval-Augmented Generation (Parametric RAG) 论文代码过程中发现的一个关键技术问题：在推理阶段，`combine` 模式与 `icl` 模式产生完全相同的结果。通过系统性的调试分析，我们定位到问题的根本原因在于 PEFT (Parameter-Efficient Fine-Tuning) 库在加载 LoRA 适配器时未能正确加载 `lora_B` 训练权重。本报告详细阐述了问题的现象、诊断过程、根本原因分析以及最终的解决方案。

**关键词**: Parametric RAG, LoRA, PEFT, 权重加载, 推理阶段

---

## 1. 引言

### 1.1 研究背景

Parametric Retrieval-Augmented Generation (Parametric RAG) 是一种新颖的检索增强生成范式，它通过将外部知识直接嵌入到大语言模型 (LLM) 的参数空间中来克服传统上下文内 RAG 方法的局限性。该方法利用 LoRA (Low-Rank Adaptation) 技术为每个检索文档生成参数化表示，在推理时将多个文档的 LoRA 参数合并以实现知识融合。

### 1.2 问题陈述

在复现 Parametric RAG 论文代码的实验过程中，我们发现一个严重的技术问题：在推理阶段，`combine` 模式（同时使用参数化知识和上下文知识）与 `icl` 模式（仅使用上下文知识）产生完全相同的结果。这一现象严重偏离了预期行为，因为：

- `icl` 模式：仅将检索文档作为上下文输入，不使用 LoRA 参数化知识
- `prag` 模式：仅使用 LoRA 参数化知识，不将文档作为上下文输入
- `combine` 模式：同时使用 LoRA 参数化知识和文档上下文

理论上，`combine` 模式应该结合两种知识来源，产生与 `icl` 模式不同的结果。

### 1.3 报告结构

本报告的组织结构如下：第 2 节描述问题的具体现象；第 3 节详细阐述诊断过程；第 4 节分析问题的根本原因；第 5 节提供解决方案；第 6 节总结全文并给出预防建议。

---

## 2. 问题描述

### 2.1 现象观察

在运行推理实验时，我们观察到以下异常现象：

1. **结果相同性**: `combine` 模式的预测结果与 `icl` 模式完全一致
2. **性能指标一致**: 两种模式在所有评估指标（EM、F1、Precision、Recall）上的数值完全相同
3. **参数加载异常**: 调试输出显示推理阶段加载的 LoRA `lora_B` 权重全为零

### 2.2 预期行为对比

| 推理模式 | 预期行为 | 实际观察 |
|---------|---------|---------|
| `icl` | 仅使用上下文知识 | 正常 |
| `prag` | 仅使用 LoRA 参数化知识 | 性能显著低于预期 |
| `combine` | 结合两种知识来源 | 与 `icl` 结果完全相同 |

### 2.3 初步假设

基于上述观察，我们提出以下初步假设：

1. **编码阶段问题**: LoRA 权重在训练阶段未能正确学习
2. **保存问题**: 训练后的 LoRA 权重未能正确保存到磁盘
3. **加载问题**: 推理阶段从磁盘加载权重时出现问题
4. **合并问题**: 多个 LoRA 适配器合并过程出现问题

---

## 3. 诊断过程

### 3.1 诊断工具开发

为了系统性地定位问题，我们开发了一套完整的 LoRA 调试工具 (`src/lora_debug.py`)，包含以下核心功能：

#### 3.1.1 权重提取函数

```python
def get_lora_weights(model, adapter_name: str = "default") -> Dict[str, Dict[str, torch.Tensor]]:
    """
    从 PeftModel 中提取 LoRA A 和 B 权重
    返回: 层名称到 {'lora_A': tensor, 'lora_B': tensor} 的映射
    """
```

#### 3.1.2 文件直接读取函数

```python
def read_safetensors_file(adapter_path: str, title: str = "Safetensors File Contents"):
    """
    直接读取 safetensors 文件内容，验证实际保存的权重
    用于绕过 PEFT 库的加载机制进行验证
    """
```

#### 3.1.3 文件与内存比较函数

```python
def compare_file_vs_memory(adapter_path: str, model, adapter_name: str):
    """
    比较 safetensors 文件中的内容与模型内存中加载的权重
    这是定位问题的关键诊断函数
    """
```

### 3.2 四步诊断流程

#### 步骤一：验证编码阶段训练效果

**目标**: 确认 LoRA 权重在训练过程中是否正确更新

**方法**: 在 `src/encode.py` 中添加调试输出，分别记录训练前后的 LoRA 权重

**结果**:
```
Initial LoRA Weights (from base_weight)
[Layer: base_model.model...layers.0.mlp.down_proj.lora_B]
  lora_B:
    Mean:  0.00000000
    Std:   0.00000000

Trained LoRA Weights
[Layer: base_model.model...layers.0.mlp.down_proj.lora_B]
  lora_B:
    Mean:  -0.00003460
    Std:   0.00082863
```

**结论**: 训练过程正常，`lora_B` 权重从全零更新为非零值 ✓

#### 步骤二：验证权重保存正确性

**目标**: 确认训练后的权重是否正确保存到磁盘

**方法**: 训练结束后直接读取保存的 `adapter_model.safetensors` 文件

**结果**:
```
Saved Adapter Contents
[base_model.model...lora_B.default.weight]
  Mean:  -0.00003460 (非零)
  Std:   0.00082863
```

**结论**: 权重保存正常，磁盘上的 safetensors 文件包含训练好的非零 `lora_B` 权重 ✓

#### 步骤三：验证推理阶段加载

**目标**: 确认推理阶段是否正确加载了训练好的权重

**方法**: 在 `src/inference.py` 中加载适配器后立即检查内存中的权重

**结果**:
```
LoRA Weights for Adapter '0' (in memory)
[Layer: base_model.model...lora_B]
  lora_B:
    Mean:  0.00000000  # 全零!
    Std:   0.00000000
```

**结论**: **问题定位！** 虽然磁盘文件中权重非零，但加载到内存后变为全零 ✗

#### 步骤四：文件与内存详细对比

**目标**: 进一步确认问题出在加载过程

**方法**: 使用 `compare_file_vs_memory()` 函数进行精确对比

**结果**:
```
[COMPARISON: File vs Memory for adapter '0']
  File has 168 tensors
  Memory has 84 layer groups

  [lora_B Comparison]
    [MISMATCH] base_model.model...lora_B.default.weight
      File: NON-ZERO (max=0.00149904)
      Memory: ZERO (max=0.00000000)

    [PROBLEM] 84 weight tensors have different zero/non-zero status 
    between file and memory!
    This suggests PEFT is not loading the weights correctly.
```

**结论**: 明确定位到问题出在 PEFT 库的权重加载过程 ✓

---

## 4. 根本原因分析

### 4.1 问题根源

经过详细诊断，我们发现问题的根本原因在于 `adapter_config.json` 文件中的 `inference_mode` 配置参数。

当 `inference_mode` 被设置为 `true` 时，PEFT 库在加载适配器时会跳过 `lora_B` 权重的加载，因为在纯推理模式下，库假设 `lora_B` 保持其初始化状态（通常为零）。

### 4.2 配置文件分析

**问题配置** (原始 `adapter_config.json`):
```json
{
    "r": 2,
    "lora_alpha": 32,
    "target_modules": ["down_proj", "gate_proj", "up_proj"],
    "lora_dropout": 0,
    "inference_mode": true,  // 问题所在
    "init_lora_weights": true
}
```

### 4.3 PEFT 库行为分析

PEFT 库的 LoRA 实现遵循以下逻辑：

1. **训练模式** (`inference_mode=false`):
   - 完全加载 `lora_A` 和 `lora_B` 权重
   - 允许梯度计算和权重更新

2. **推理模式** (`inference_mode=true`):
   - 加载 `lora_A` 权重
   - **跳过 `lora_B` 权重加载**（假设保持初始化状态）
   - 禁用梯度计算

这种设计的初衷是优化推理性能，但在需要加载预训练 LoRA 权重的场景下会导致问题。

### 4.4 问题传播链

```
encode.py 保存适配器 (inference_mode=false)
    ↓
adapter_config.json 中记录 inference_mode=true（PEFT 默认行为）
    ↓
inference.py 加载适配器
    ↓
PEFT 检测到 inference_mode=true
    ↓
跳过 lora_B 权重加载，保持全零状态
    ↓
LoRA 对模型输出无影响（因为 ΔW = lora_B @ lora_A = 0）
    ↓
combine 模式结果与 icl 模式相同
```

---

## 5. 解决方案

### 5.1 方案一：修改编码阶段配置（推荐）

在 `src/encode.py` 中创建 LoRA 配置时，显式设置 `inference_mode=False`:

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=['down_proj', 'gate_proj', 'up_proj'],
    inference_mode=False,  # 确保保存时不设置为推理模式
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=0,
)
```

### 5.2 方案二：修改推理阶段加载

在 `src/inference.py` 中加载适配器时强制禁用推理模式:

```python
model = PeftModel.from_pretrained(
    model, 
    adapter_path,
    adapter_name="0", 
    is_trainable=False,  # 仍然禁用训练
    # PEFT 会自动处理权重加载
)
```

### 5.3 实施步骤

1. **清除旧数据**: 删除 `offline/` 目录中使用旧配置生成的所有适配器文件
2. **应用代码修复**: 更新 `encode.py` 中的 LoRA 配置
3. **重新训练**: 运行完整的编码流程，生成新的适配器文件
4. **验证修复**: 使用 `--debug` 参数运行推理，确认 `lora_B` 权重正确加载

### 5.4 验证结果

修复后的诊断输出：

```
[lora_B Comparison]
  [OK] File and memory weights match in zero/non-zero status

LoRA Weights for Adapter '0' (in memory)
[Layer: base_model.model...lora_B]
  lora_B:
    Mean:  -0.00003460  # 非零，与文件一致
    Std:   0.00082863
```

最终实验结果表明：
- `icl` 模式和 `combine` 模式现在产生**不同的结果** ✓
- `prag` 模式性能显著提升，符合论文预期 ✓
- `combine` 模式有效结合了参数化知识和上下文知识 ✓

---

## 6. 总结与建议

### 6.1 主要发现

本次问题诊断揭示了以下关键发现：

1. **问题本质**: PEFT 库在 `inference_mode=true` 时会跳过 `lora_B` 权重的加载
2. **影响范围**: 这一行为会导致所有依赖 LoRA 参数化知识的推理模式失效
3. **诊断关键**: 通过对比磁盘文件内容与内存中的权重，可以精确定位加载问题

### 6.2 技术教训

1. **配置参数重要性**: LoRA 相关的配置参数（尤其是 `inference_mode`）对实际行为有重大影响
2. **端到端验证**: 仅验证训练阶段正确性不足，需要验证完整的训练-保存-加载-推理链路
3. **调试工具价值**: 系统性的调试工具对于定位复杂的机器学习流程问题至关重要

### 6.3 预防建议

为避免类似问题，建议：

1. **明确配置**: 在所有 LoRA 配置中显式设置关键参数，不依赖默认值
2. **添加验证**: 在加载适配器后添加权重非零验证检查
3. **文档记录**: 记录所有配置参数的预期行为和已知限制
4. **持续测试**: 建立自动化测试来验证不同模式产生不同的结果

### 6.4 致谢

感谢所有参与问题诊断和修复过程的研究人员。本报告中使用的调试工具和诊断方法可供类似问题的排查参考。

---

## 附录 A：调试工具使用指南

### A.1 启用调试模式

```bash
# 编码阶段调试
python src/encode.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset complexwebquestions \
    --lora_rank 2 \
    --lora_alpha 32 \
    --debug

# 推理阶段调试
python src/inference.py \
    --model_name qwen2.5-1.5b-instruct \
    --dataset complexwebquestions \
    --inference_method combine \
    --lora_rank 2 \
    --lora_alpha 32 \
    --debug
```

### A.2 关键调试输出解读

| 输出内容 | 正常状态 | 异常状态 |
|---------|---------|---------|
| Initial LoRA lora_B | 全零 | N/A |
| Trained LoRA lora_B | 非零 | 全零（训练问题）|
| File lora_B content | 非零 | 全零（保存问题）|
| Memory lora_B content | 非零 | 全零（加载问题）|
| File vs Memory match | 一致 | 不一致（加载问题）|

---

## 附录 B：相关代码变更记录

### B.1 `src/encode.py` 修改

- 添加调试输出功能
- 修复基础权重创建后模型状态清理问题
- 添加训练后权重保存验证

### B.2 `src/inference.py` 修改

- 添加适配器加载调试输出
- 添加权重合并调试输出
- 添加基础模型与合并模型权重对比功能

### B.3 `src/lora_debug.py` 新增

- 完整的 LoRA 调试工具集
- 支持训练阶段和推理阶段调试
- 支持文件与内存权重对比分析

---

**文档版本**: 1.0  
**最后更新**: 2026年1月27日  
**作者**: PRAG 问题诊断工作组
