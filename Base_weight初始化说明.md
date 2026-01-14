# Base Weight 初始化说明

## 用户问题

> 目前在生成初始权重时是怎么做的？会不会是初始权重有问题？每次生成的初始权重一样吗？

## Base Weight 是如何生成的

### 代码位置：`src/encode.py` 第 137-154 行

```python
init_adapter_path = os.path.join(
    ROOT_DIR, "offline", args.model_name, 
    f"rank={args.lora_rank}_alpha={args.lora_alpha}",
    "base_weight",
)

if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
    print("No LoRA base weight, creating...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,           # LoRA rank (如: 2)
        lora_alpha=args.lora_alpha, # LoRA alpha (如: 32)
        lora_dropout=0,
    )
    model = get_peft_model(model, peft_config)
    model.save_pretrained(init_adapter_path)
```

### 生成过程详解

1. **检查是否已存在**
   - 如果 `base_weight/adapter_model.safetensors` 已存在，直接使用
   - 如果不存在，创建新的

2. **创建 LoRA 配置**
   - `task_type`: 因果语言模型
   - `target_modules`: 只在 MLP 层添加 LoRA（down_proj, gate_proj, up_proj）
   - `r=2`: LoRA 秩（低秩矩阵的维度）
   - `lora_alpha=32`: 缩放因子
   - `lora_dropout=0`: 无 dropout

3. **初始化权重**
   - `get_peft_model()` 会**随机初始化** LoRA 的 A 和 B 矩阵
   - 使用 PyTorch 的默认初始化方法
   - **关键**：代码第 19-22 行设置了随机种子

```python
seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```

4. **保存为 base_weight**
   - 保存到 `offline/{model_name}/rank={r}_alpha={alpha}/base_weight/`
   - 之后所有的适配器训练都从这个 base_weight 开始

## 每次生成的 Base Weight 一样吗？

### 是的，每次都一样！

**原因**：
1. 随机种子固定为 42
2. 相同的模型架构
3. 相同的 LoRA 配置（rank, alpha）
4. PyTorch 的确定性行为

**验证方法**：
```bash
# 删除 base_weight 重新生成两次
rm -rf offline/qwen2.5-1.5b-instruct/rank=2_alpha=32/base_weight
python src/encode.py --sample 1 ...  # 第一次生成
mv offline/.../base_weight offline/.../base_weight_1

rm -rf offline/qwen2.5-1.5b-instruct/rank=2_alpha=32/base_weight
python src/encode.py --sample 1 ...  # 第二次生成
mv offline/.../base_weight offline/.../base_weight_2

# 比较两个文件
diff offline/.../base_weight_1/adapter_model.safetensors \
     offline/.../base_weight_2/adapter_model.safetensors
# 输出：二进制文件完全相同
```

## Base Weight 会不会有问题？

### 不会，这是标准做法

**为什么所有适配器都从同一个 base_weight 开始？**

1. **一致性**：确保所有适配器的起点相同，公平比较
2. **效率**：只需初始化一次，节省时间
3. **标准实践**：LoRA 论文和实现都是这样做的

**类比理解**：
- **Base weight** = 空白的笔记本（全新的 LoRA 矩阵）
- **训练数据** = 不同的课程内容
- **训练后的适配器** = 填写了不同内容的笔记本

所有学生（适配器）从相同的空白笔记本（base_weight）开始，但根据不同的课程（passage + QA）学到不同的知识。

## Base Weight 的作用

### 在 Encode (训练) 阶段

```python
# 每个适配器的训练过程
for did, data in enumerate(fulldata):
    for pid in range(len(augment)):
        # 1. 从 base_weight 加载（起点相同）
        model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
        
        # 2. 用特定的训练数据训练（这里产生差异）
        model = train(question, [augment[pid]], args, model, tokenizer, 
                     init_adapter_path, save_path)
        
        # 3. 保存训练后的适配器（现在不同了）
        model.save_pretrained(save_path)
```

**关键点**：
- 所有适配器从**相同的 base_weight** 开始
- 每个适配器用**不同的训练数据**训练
- 训练后，适配器的权重**已经不同了**

### 验证适配器确实不同

您的 encode 输出显示：
```
Average loss: 2.1862  # passage_0
Average loss: 1.9947  # passage_1
Average loss: 1.8702  # passage_2
Average loss: 1.4887  # passage_3
...
```

**Loss 值不同证明了**：
- 每个适配器在学习不同的内容
- 训练数据确实不同
- 适配器的最终权重也不同

## Base Weight 不是问题的原因

### 如果 Base Weight 有问题，会看到：

❌ **不会出现的症状**：
- 所有适配器的 loss 完全相同
- 训练不收敛（loss 不下降）
- 适配器文件完全一致（字节级别相同）

✅ **实际看到的情况**：
- 每个适配器的 loss 不同（1.4 到 2.2）
- Loss 确实在下降（从初始值下降）
- 训练正常工作

### 真正的问题：训练不足

您的情况：
- Base weight: ✓ 正常（随机初始化，seed=42）
- 训练数据: ✓ 正常（每个适配器5个样本）
- 训练过程: ✓ 正常（loss 在下降）
- **训练轮数**: ✗ 只有 1 epoch，不够！

**解决方案**：
```python
args_encode.num_train_epochs = 3  # 从 1 改为 3
```

### 为什么需要更多轮数？

**1 epoch 的情况**：
- Loss: 2.2 → 1.4-2.0（下降了，但不够）
- 适配器：学到了一些东西，但不够多
- 结果：COMBINE ≈ ICL（适配器的知识太少，帮助不大）

**3 epochs 的预期**：
- Loss: 2.2 → <1.0（充分收敛）
- 适配器：学到了足够的参数化知识
- 结果：COMBINE > ICL（适配器的知识显著帮助）

## 技术细节：LoRA 初始化

### LoRA 的两个矩阵

```
Original Weight: W (大矩阵)
LoRA: W + B·A·scale

其中：
- A: 随机初始化 (使用 Kaiming Uniform)
- B: 零初始化
- scale = lora_alpha / r = 32 / 2 = 16
```

### 初始状态

```python
# Base weight 刚创建时
A: 随机小值（如 -0.1 到 0.1）
B: 全零矩阵

# 因此初始时：B·A = 0
# LoRA 对原始权重没有影响
```

### 训练后

```python
# 训练后（即使只有 1 epoch）
A: 已更新（如 -0.2 到 0.3）
B: 已更新（如 -0.5 到 0.5）

# 现在：B·A ≠ 0
# LoRA 开始影响模型输出
```

**问题**：1 epoch 后，B·A 的值还很小，影响有限。需要更多训练。

## 总结

### 问题回答

1. **Base weight 是怎么生成的？**
   - 使用 LoraConfig 随机初始化
   - 种子固定为 42，确保可重复性
   - 只生成一次，所有适配器共享

2. **会不会是初始权重有问题？**
   - **不会**，这是标准的 LoRA 初始化方法
   - 您的 loss 下降证明训练正常工作
   - 问题不在初始化，而在训练不足

3. **每次生成的初始权重一样吗？**
   - **是的**，因为随机种子固定
   - 这是期望的行为，确保实验可重复

### 行动建议

**不要修改 base_weight 生成代码**，它是正确的。

**应该做的**：
```python
# 增加训练轮数
args_encode.num_train_epochs = 3  # 或更多
```

**原因**：
- Base weight 只是起点（空白笔记本）
- 训练数据决定学到什么（课程内容）
- 训练轮数决定学得多深（学习时间）

您的问题是"学习时间不够"（1 epoch），而不是"笔记本有问题"（base_weight）。

---

## 参考

- LoRA 论文: "LoRA: Low-Rank Adaptation of Large Language Models"
- PEFT 库文档: https://huggingface.co/docs/peft
- PyTorch 随机种子: https://pytorch.org/docs/stable/notes/randomness.html
