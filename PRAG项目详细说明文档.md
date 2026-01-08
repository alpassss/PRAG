# Parametric RAG (PRAG) 项目详细说明文档

## 一、论文核心思想

### 1.1 论文背景

**论文标题**: Parametric Retrieval Augmented Generation  
**发表会议**: SIGIR 2025  
**论文链接**: https://arxiv.org/abs/2501.15915

### 1.2 核心创新点

传统的检索增强生成（RAG）系统存在以下局限：

1. **计算开销大**: 需要将检索到的文档附加到输入上下文中，导致上下文长度增加，消耗大量计算资源
2. **知识整合差距**: 上下文注入只影响输入，而LLM的核心知识存储在其参数中，存在本质的整合鸿沟

**Parametric RAG的解决方案**：

将外部知识**直接参数化**到大语言模型的前馈网络（FFN）参数中，通过LoRA（Low-Rank Adaptation）技术将文档知识"记忆"到模型参数空间，从而：

- **减少推理时计算成本**（无需每次推理都附加文档文本）
- **提升推理和知识合成能力**（知识深度整合到模型参数中）
- **可与上下文RAG结合**（获得最佳性能）

实验结果显示，Parametric RAG在推理速度上提升30%，知识存储效率提升40%。

### 1.3 方法论概述

Parametric RAG包含三个核心步骤：

1. **自增强（Self-Augmentation）**: 将文档转换为增强数据集
   - 文档改写：保持实体和关键信息不变的情况下重写文档
   - QA生成：为每个文档生成3个问答对

2. **参数训练（Parameter Training）**: 通过LoRA训练额外参数
   - 为每个检索到的文档训练独立的LoRA参数
   - 参数编码文档的知识内容

3. **推理（Inference）**: 合并相关文档的参数并插入LLM
   - 将相关文档的LoRA参数合并
   - 使用更新后的LLM进行推理

---

## 二、仓库文件结构分析

### 2.1 根目录文件

```
PRAG/
├── README.md                    # 项目说明文档（英文）
├── all_prompt.md               # 所有实验使用的提示词模板
├── requirements.txt            # Python依赖包列表
├── prep_elastic.py             # Elasticsearch索引准备脚本
├── data_aug.tar.gz             # 预处理好的增强数据压缩包
├── assets/                     # 图片资源文件夹
├── configs/                    # 实验配置脚本
├── src/                        # 源代码目录
└── .git/                       # Git版本控制
```

### 2.2 源代码目录 (src/)

#### 核心文件

| 文件名 | 作用 | 对应论文章节 |
|--------|------|-------------|
| `augment.py` | 数据增强模块 | Section 3.2.1 Self-Augmentation |
| `encode.py` | 文档参数化训练 | Section 3.2.2 Additional Parameter Training |
| `inference.py` | 推理生成模块 | Section 3.3 Inference |
| `utils.py` | 工具函数集合 | - |
| `prompt_template.py` | 提示词模板管理 | - |
| `root_dir_path.py` | 根目录路径配置 | - |
| `warmup_lora.py` | LoRA预热训练 | - |
| `get_warmup_data.py` | 生成预热训练数据 | - |

#### 子目录

**retrieve/** - 检索模块
- `retriever.py`: BM25检索器实现
- `beir/`: BEIR检索评估框架

**fewshot/** - Few-shot样例
- `2wikimultihopqa.json`: 2WikiMultihopQA数据集的few-shot示例
- `hotpotqa.json`: HotpotQA数据集的few-shot示例

### 2.3 配置文件目录 (configs/)

包含12个shell脚本，对应不同数据集和模型的组合：

| 数据集 | 模型 |
|--------|------|
| 2wikimultihopqa | llama3.2-1b-instruct |
| 2wikimultihopqa | llama3-8b-instruct |
| 2wikimultihopqa | qwen2.5-1.5b-instruct |
| hotpotqa | llama3.2-1b-instruct |
| hotpotqa | llama3-8b-instruct |
| hotpotqa | qwen2.5-1.5b-instruct |
| popqa | llama3.2-1b-instruct |
| popqa | llama3-8b-instruct |
| popqa | qwen2.5-1.5b-instruct |
| complexwebquestions | llama3.2-1b-instruct |
| complexwebquestions | llama3-8b-instruct |
| complexwebquestions | qwen2.5-1.5b-instruct |

每个配置文件包含两个主要命令：
1. `encode.py` - 文档参数化训练命令
2. `inference.py` - 推理生成命令

---

## 三、完整复现步骤详解

### 3.1 环境准备

#### 3.1.1 创建Conda环境

```bash
# 创建Python 3.10.4环境
conda create -n prag python=3.10.4

# 激活环境
conda activate prag
```

**为什么选择Python 3.10.4**：
- 与transformers 4.44.2和torch 1.13.1兼容性最好
- 避免新版本Python可能存在的依赖冲突

#### 3.1.2 安装依赖包

```bash
# 先安装PyTorch
pip install torch==2.1.0

# 安装其他依赖
pip install -r requirements.txt
```

**requirements.txt内容解析**：
```
torch==1.13.1              # 深度学习框架（注意：实际使用2.1.0）
transformers==4.44.2       # Hugging Face模型库
elasticsearch==8.15.0      # 文档检索后端
peft==0.13.2              # Parameter-Efficient Fine-Tuning库（用于LoRA）
pandas==1.5.3             # 数据处理
numpy==1.26.4             # 数值计算
faiss-cpu==1.8.0          # 向量检索（可选，主要用BM25）
termcolor                 # 终端彩色输出
```

#### 3.1.3 配置根目录路径

```bash
# 编辑src/root_dir_path.py文件
vim src/root_dir_path.py
```

将`ROOT_DIR = "path_to_PRAG"`修改为实际路径，例如：
```python
ROOT_DIR = "/home/user/PRAG"
```

**为什么需要这一步**：
- 代码中多处使用绝对路径引用数据和输出
- 统一管理路径便于在不同机器上部署

### 3.2 数据准备

#### 3.2.1 方法一：使用预处理数据（推荐）

```bash
# 解压预处理的增强数据
tar -xzvf data_aug.tar.gz
```

解压后会生成`data_aug/`目录，包含四个数据集的增强数据：
- `data_aug/2wikimultihopqa/`
- `data_aug/hotpotqa/`
- `data_aug/popqa/`
- `data_aug/complexwebquestions/`

每个子目录包含不同模型生成的增强数据文件。

#### 3.2.2 方法二：从头开始准备数据

##### 步骤1：下载Wikipedia知识库

```bash
# 创建数据目录
mkdir -p data/dpr

# 下载DPR预处理的Wikipedia数据
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz

# 解压
cd data/dpr
gzip -d psgs_w100.tsv.gz
cd ../..
```

**数据说明**：
- 文件大小：约13GB（压缩后约3.8GB）
- 格式：TSV（Tab-Separated Values）
- 内容：2100万个Wikipedia段落
- 字段：`id, text, title`

##### 步骤2：安装和启动Elasticsearch

```bash
# 进入data目录
cd data

# 下载Elasticsearch 8.15.0
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz

# 解压
tar zxvf elasticsearch-8.15.0.tar.gz

# 删除压缩包
rm elasticsearch-8.15.0.tar.gz

# 进入Elasticsearch目录
cd elasticsearch-8.15.0

# 后台启动Elasticsearch
nohup bin/elasticsearch &

# 返回项目根目录
cd ../..
```

**Elasticsearch启动说明**：
- 默认端口：9200
- 启动时间：约30-60秒
- 内存需求：至少4GB可用内存

**验证Elasticsearch是否启动成功**：
```bash
# 等待30秒后检查
sleep 30
curl http://localhost:9200

# 成功会返回JSON响应
```

##### 步骤3：构建BM25索引

```bash
# 运行索引构建脚本
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

**索引构建过程**：
- 时间：约30-60分钟（取决于机器性能）
- 内存占用：峰值可达8-10GB
- 索引大小：约15-20GB磁盘空间

**prep_elastic.py的作用**：
1. 读取psgs_w100.tsv文件
2. 为每个段落创建Elasticsearch文档
3. 批量插入到"wiki"索引中
4. 建立BM25倒排索引

##### 步骤4：下载QA数据集

**2WikiMultihopQA**:
```bash
# 手动下载
# 访问: https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip
# 下载并解压到 data/2wikimultihopqa/

# 或使用命令行（需要dropbox工具）
mkdir -p data/2wikimultihopqa
# 将下载的文件解压到该目录
```

数据集文件：
- `dev.json`: 开发集（12576个问题）
- `train.json`: 训练集
- `id_aliases.json`: 实体别名映射

**HotpotQA**:
```bash
mkdir -p data/hotpotqa
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

数据集特点：
- 开发集：7405个问题
- 类型：bridge（桥接）和comparison（比较）问题
- 包含金标准支撑事实

**PopQA**:
```bash
mkdir -p data/popqa
wget -P data/popqa https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv
```

数据集特点：
- 14267个流行实体问题
- TSV格式，包含问题、答案和别名

**ComplexWebQuestions**:
```bash
# 手动下载
# 访问: https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&e=1
# 下载 ComplexWebQuestions_dev.json 到 data/complexwebquestions/

mkdir -p data/complexwebquestions
# 将下载的文件放到该目录
```

数据集特点：
- 3519个复杂问题
- 需要多步推理

##### 步骤5：执行数据增强

```bash
# 示例：对2WikiMultihopQA数据集进行增强
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300 \
    --topk 3
```

**参数说明**：
- `--model_name`: 用于增强的模型（llama3.2-1b-instruct / qwen2.5-1.5b-instruct / llama3-8b-instruct）
- `--dataset`: 数据集名称
- `--data_path`: 原始数据集路径
- `--sample`: 处理的问题数量（300表示处理前300个问题）
- `--topk`: 为每个问题检索的文档数量

**增强过程详解**：

1. **加载数据集**（第209-220行）：
   - 调用相应的`load_{dataset}`函数加载原始数据
   - 如果有多个子类型，分别处理

2. **初始化模型**（第222-229行）：
   - 加载指定的LLM和tokenizer
   - 配置生成参数（max_new_tokens=512, temperature=0.7）

3. **对每个问题进行处理**（第240-259行）：
   
   a. **BM25检索**（第241行）：
   ```python
   passages = bm25_retrieve(data["question"], topk=args.topk+10)
   ```
   - 使用问题检索top-k+10个相关文档
   - 多检索几个作为备选，防止某些文档增强失败

   b. **文档改写**（第248行）：
   ```python
   rewrite = get_rewrite(passage, model_name, model, tokenizer, generation_config)
   ```
   - 提示词见all_prompt.md中的"Document Rewriting"
   - 保持实体和关键信息不变，用不同方式表达
   - 目的：增加数据多样性，提高模型鲁棒性

   c. **QA生成**（第250行）：
   ```python
   qa = get_qa(passage, model_name, model, tokenizer, generation_config)
   ```
   - 为每个文档生成3个问答对
   - 每个QA包含：question, answer, full_answer
   - 验证生成的QA是否符合格式要求

   d. **数据验证**（第251行）：
   ```python
   if fix_qa(qa)[0] == False:  # 跳过错误的文档
       continue
   ```
   - 检查QA格式是否正确
   - 如果格式错误，跳过这个文档，继续下一个

4. **保存增强数据**（第261-262行）：
   - 输出到`data_aug/{dataset}/{model_name}/{filename}.json`
   - JSON格式，便于后续加载

**增强数据格式**：
```json
[
    {
        "test_id": 0,
        "question": "原始问题",
        "answer": "答案",
        "passages": ["passage1", "passage2", "passage3"],
        "augment": [
            {
                "pid": 0,
                "passage": "原始文档",
                "llama3.2-1b-instruct_rewrite": "改写后的文档",
                "llama3.2-1b-instruct_qa": [
                    {
                        "question": "生成的问题1",
                        "answer": "答案1",
                        "full_answer": "完整答案1"
                    },
                    ...
                ]
            },
            ...
        ]
    },
    ...
]
```

**为什么要进行数据增强**：
1. **文档改写**：提供同一知识的不同表述，增强模型的泛化能力
2. **QA生成**：为训练提供更多监督信号，每个文档不仅包含原文，还包含显式的问答对
3. **多样性**：一半的QA使用原文档，一半使用改写文档，一半仅使用问题-答案（见encode.py第78-90行）

**估计时间和资源**：
- 使用llama3.2-1b-instruct，300个问题，topk=3：约2-3小时
- 使用llama3-8b-instruct：约6-8小时
- GPU内存需求：
  - 1B模型：至少6GB
  - 8B模型：至少24GB

### 3.3 LoRA基础权重预热（可选但推荐）

#### 3.3.1 为什么需要预热

**不预热的情况**：
- LoRA从随机初始化开始
- 每个文档的LoRA可能学到重复的基础能力
- 训练效率低

**预热的好处**：
- LoRA从一个已经具备基础QA能力的权重开始
- 训练更快收敛
- 每个文档的LoRA专注于文档特定知识

#### 3.3.2 生成预热数据

```bash
python src/get_warmup_data.py
```

**这个脚本做什么**（分析get_warmup_data.py）：

1. **create_direct()**（第18-32行）：
   - 为PopQA和ComplexWebQuestions创建直接问答数据
   - 从数据集的**后半部分**提取（第21行：`dataset[1000:]`）
   - 目的：避免数据泄露，因为前300个用于测试
   - 输出到`warmup/data/direct/{dataset}.json`

2. **create_cot()**（第114-186行）：
   - 为2WikiMultihopQA和HotpotQA创建带有Chain-of-Thought的数据
   - 使用llama3-8b-instruct生成推理过程
   - 只保留生成正确答案的样本（第173行）
   - 每个数据集生成300个样本
   - 输出到`warmup/data/cot/train_data.json`

**数据防泄露机制**：
- 找到每种类型的前300个问题的最后一个索引
- 从该索引之后1000个位置开始采样
- 确保预热数据与测试数据完全不重叠

#### 3.3.3 训练预热权重

**训练不带CoT的基础权重**（用于PopQA和ComplexWebQuestions）：
```bash
python src/warmup_lora.py \
    --model_name llama3.2-1b-instruct \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --block_size 3000 \
    --lora_rank 2 \
    --lora_alpha 32
```

**训练带CoT的基础权重**（用于2WikiMultihopQA和HotpotQA）：
```bash
python src/warmup_lora.py \
    --model_name llama3.2-1b-instruct \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --block_size 3000 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot
```

**训练过程分析**（warmup_lora.py）：

1. **加载训练数据**（第92-115行）：
   - 不带CoT：从多个数据集混合采样，每个最多1000条
   - 带CoT：使用预生成的CoT数据

2. **配置LoRA**（第117-124行）：
   ```python
   target_modules=['down_proj', 'gate_proj', 'up_proj']
   ```
   - 只在FFN的三个投影层上添加LoRA
   - 不修改attention层（保持原始语义理解能力）

3. **训练循环**（第141-150行）：
   - 使用AdamW优化器
   - 每10步打印一次loss
   - 记录loss曲线

4. **保存权重**（第151-167行）：
   - 保存到`warmup/lora_base_weight/{model_name}/{cot|direct}/`
   - 同时保存训练配置和loss曲线图

**预热训练资源需求**：
- 600个数据点：约15-30分钟
- 2000个数据点：约1-2小时
- GPU内存：与模型大小相同（1B约6GB，8B约24GB）

### 3.4 文档参数化训练

这是Parametric RAG的核心步骤，将每个文档的知识编码为LoRA参数。

#### 3.4.1 使用配置脚本运行

```bash
# 示例：2WikiMultihopQA + Llama3.2-1B
bash configs/2wikimultihopqa_llama3.2-1b-instruct.sh
```

这个脚本包含两个命令：
1. `encode.py` - 文档参数化
2. `inference.py` - 推理测试

让我们分别详细分析。

#### 3.4.2 encode.py详细解析

**命令示例**：
```bash
python src/encode.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --sample 300 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot
```

**执行流程**（encode.py源码分析）：

1. **初始化阶段**（第124-155行）：

   a. 加载增强数据：
   ```python
   data_list = load_data(args.dataset, args.data_type, args.augment_model)
   ```
   - 从`data_aug/{dataset}/{augment_model}/`加载
   - 如果没指定augment_model，使用当前model_name

   b. 加载基础模型：
   ```python
   model, tokenizer, _generation_config = get_model(args.model_name)
   ```

   c. 检查或创建base_weight：
   ```python
   init_adapter_path = os.path.join(ROOT_DIR, "offline", args.model_name, 
                                    f"rank={args.lora_rank}_alpha={args.lora_alpha}",
                                    "base_weight")
   ```
   - 如果预热了，这里会使用预热的权重
   - 如果没有，创建随机初始化的LoRA（第138-154行）

2. **为每个问题训练**（第172-179行）：

   对每个问题，为其检索到的每个文档训练一个独立的LoRA：

   ```python
   for did, data in enumerate(fulldata):
       augment = data["augment"]
       for pid in range(len(augment)):
           save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
           if os.path.exists(...):  # 如果已训练，跳过
               continue
           model = train(...)  # 训练这个文档的LoRA
   ```

3. **单个文档的训练**（train函数，第94-121行）：

   a. **准备训练数据**（get_train_data函数，第71-91行）：
   ```python
   for qa in qas:
       if qid < qpa_cnt:  # 前一半QA
           # 使用原文档和改写文档
           for passage in [original, rewrite]:
               prompt_ids.append(get_prompt(..., passages=[passage], ...))
       else:  # 后一半QA
           # 不使用文档，只有问答
           prompt_ids.append(get_prompt(..., passages=None, ...))
   ```
   
   为什么这样设计？
   - **前一半QA + 文档**：让模型学会利用文档回答问题
   - **后一半QA - 文档**：让模型将知识记忆到参数中
   - **原文档 + 改写**：增加数据多样性

   b. **训练循环**（第109-115行）：
   ```python
   for epoch in range(args.num_train_epochs):
       for step, batch in enumerate(train_dataloader):
           outputs = model(**batch)
           loss = outputs.loss
           loss.backward()
           optimizer.step()
   ```
   - 简单的监督学习
   - 优化LoRA参数使其能预测正确答案

   c. **保存和卸载**（第116-120行）：
   ```python
   model.save_pretrained(save_path)
   model = model.unload()  # 卸载LoRA
   torch.cuda.empty_cache()
   gc.collect()
   ```
   - 保存当前文档的LoRA
   - 卸载以准备训练下一个文档的LoRA

4. **输出目录结构**：
```
offline/
└── llama3.2-1b-instruct/
    └── rank=2_alpha=32/
        ├── base_weight/              # 基础LoRA权重
        │   └── adapter_model.safetensors
        └── 2wikimultihopqa/
            └── lr=0.0003_epoch=1_cot/
                └── aug_model=llama3.2-1b-instruct/
                    └── total/
                        ├── data_0/
                        │   ├── passage_0/
                        │   │   └── adapter_model.safetensors
                        │   ├── passage_1/
                        │   └── passage_2/
                        ├── data_1/
                        └── ...
```

**关键参数说明**：

- `--lora_rank=2`：LoRA的秩
  - 更大的秩 = 更强的表达能力 = 更多参数
  - 2是较小的值，平衡效果和效率
  
- `--lora_alpha=32`：LoRA的缩放因子
  - 实际学习率 = lr * alpha / rank = 0.0003 * 32 / 2 = 0.0048
  - 较大的alpha使LoRA的影响更明显

- `--num_train_epochs=1`：训练轮数
  - 每个文档只训练1轮已足够
  - 过多训练可能导致过拟合

- `--with_cot`：是否使用Chain-of-Thought
  - 对于多跳推理任务（2Wiki, HotpotQA）使用
  - 对于简单QA（PopQA, ComplexWebQuestions）不使用

**估计时间**：
- 300个问题，每个3个文档 = 900个LoRA
- 每个LoRA训练：10-30秒
- 总计：2.5-7.5小时（取决于GPU）

### 3.5 推理和评估

#### 3.5.1 inference.py详细解析

**命令示例**：
```bash
python src/inference.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --sample 300 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --max_new_tokens 128 \
    --inference_method combine \
    --with_cot
```

**执行流程**（inference.py源码分析）：

1. **加载数据和模型**（第14-20行）：
```python
data_list = load_data(args.dataset, args.data_type, args.augment_model)
model, tokenizer, generation_config = get_model(args.model_name, max_new_tokens=args.max_new_tokens)
```

2. **确定LoRA加载路径**（第23-31行）：
```python
load_adapter_path = os.path.join(
    ROOT_DIR, "offline", args.model_name,
    f"rank={args.lora_rank}_alpha={args.lora_alpha}",
    args.dataset,
    f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
    f"aug_model={args.augment_model}",
)
```
- 路径必须与训练时完全一致
- 参数不匹配会导致找不到LoRA

3. **对每个问题进行推理**（第54-101行）：

**三种推理方法**：

a. **ICL (In-Context Learning)** - 传统RAG：
```python
if args.inference_method == "icl":
    ret.append(get_pred(model, psgs=passages))
```
- 将检索到的文档拼接到prompt中
- 模型根据上下文生成答案
- 作为baseline

b. **PRAG (Parametric RAG)** - 仅使用参数：
```python
else:  # prag or combine
    # 加载所有相关文档的LoRA
    for pid in range(len(passages)):
        adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}")
        if pid == 0:
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name="0", is_trainable=False)
        else:
            model.load_adapter(adapter_path, adapter_name=str(pid))
    
    # 合并所有LoRA
    model.add_weighted_adapter(
        adapters=[str(i) for i in range(len(passages))],
        weights=[1] * len(passages),  # 等权重合并
        adapter_name="merge",
        combination_type="cat",  # 拼接合并
    )
    model.set_adapter("merge")
    
    # 推理（不使用上下文）
    ret.append(get_pred(model, psgs=None))
```

c. **Combine** - 参数 + 上下文：
```python
# 与PRAG相同的加载和合并过程
# 但推理时同时提供上下文
ret.append(get_pred(model, psgs=passages))
```

**LoRA合并机制**：
- `combination_type="cat"`：拼接合并
- 假设每个LoRA的rank=2，有3个文档
- 合并后的LoRA rank=6（2×3）
- 每个文档的知识被编码为独立的子空间

**为什么等权重合并**：
- 简单有效
- BM25已经按相关性排序
- 未来可以探索加权合并（如按BM25分数）

4. **清理和准备下一个问题**（第98-101行）：
```python
model.delete_adapter("merge")
model = model.unload()
torch.cuda.empty_cache()
gc.collect()
```
- 删除合并的adapter
- 卸载所有LoRA
- 释放GPU内存

5. **评估**（第106-115行）：
```python
metrics = ["em", "f1", "prec", "recall"]
for met in metrics:
    acc = sum(float(d[met]) for d in ret) / len(ret)
```

**评估指标**：
- **EM (Exact Match)**：预测答案完全匹配标准答案
- **F1**：token级别的F1分数
- **Precision**：预测答案中正确token的比例
- **Recall**：标准答案中被预测到的token比例

**答案提取**（utils.py evaluate函数，第185-218行）：
- 不带CoT：提取第一个句子
- 带CoT：提取"the answer is"之后的内容

6. **输出目录结构**：
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
                            └── result.txt       # 总体评估指标
```

**predict.json格式**：
```json
[
    {
        "test_id": 0,
        "question": "问题",
        "answer": "标准答案",
        "text": "模型生成的文本",
        "eval_predict": "提取的答案",
        "em": "1",
        "f1": "1.0",
        "prec": "1.0",
        "recall": "1.0"
    },
    ...
]
```

**result.txt格式**：
```
em      0.7234
f1      0.8123
prec    0.8234
recall  0.8012

{
    "model_name": "llama3.2-1b-instruct",
    "dataset": "2wikimultihopqa",
    ...
}
```

#### 3.5.2 三种推理方法的对比

| 方法 | 使用文档上下文 | 使用文档参数 | 优点 | 缺点 |
|------|--------------|------------|------|------|
| ICL | ✓ | ✗ | 无需训练，灵活 | 计算开销大，上下文长度限制 |
| PRAG | ✗ | ✓ | 推理快，无长度限制 | 需要训练，固定知识 |
| Combine | ✓ | ✓ | 性能最好 | 计算开销较大 |

**论文实验结果**（Figure 2）：
- PRAG相比ICL：推理速度提升30%
- Combine相比ICL：准确率提升10-15%
- PRAG相比ICL：在大多数任务上准确率相当或更好

---

## 四、深入理解关键技术点

### 4.1 为什么选择FFN层添加LoRA

**FFN在Transformer中的作用**：
```
input -> LayerNorm -> Attention -> Residual
                                     ↓
                                  LayerNorm -> FFN -> Residual
                                                       ↓
                                                    output
```

FFN结构：
```
FFN(x) = gelu(x @ W_gate) ⊙ (x @ W_up) @ W_down
```

**为什么FFN存储知识**：
- 研究表明FFN像一个键值记忆库
- W_up和W_gate选择相关的"记忆"
- W_down提取和组合记忆
- Attention主要负责语义理解和推理

**只修改FFN的好处**：
- 保持原始的语义理解能力
- 只添加新的事实知识
- 避免破坏模型的推理能力

### 4.2 LoRA参数量分析

**原始FFN参数**（以Llama3.2-1B为例）：
- hidden_size = 2048
- intermediate_size = 8192
- FFN参数 = 2048×8192 + 8192×2048 + 2048×8192 ≈ 50M per layer
- 16层 × 50M = 800M参数

**LoRA参数**（rank=2）：
- 每个投影：2048×2 + 2×8192 = 20k参数
- 三个投影（up, down, gate）：60k per layer
- 16层 × 60k = 960k ≈ 1M参数

**压缩比**：800M / 1M = 800x

**存储需求**（每个文档的LoRA）：
- float32: 1M × 4 bytes = 4MB
- bfloat16: 1M × 2 bytes = 2MB

### 4.3 LoRA合并的数学原理

**原始FFN**：
```
y = x @ W_down
```

**单个LoRA**：
```
y = x @ (W_down + A @ B)
  = x @ W_down + x @ A @ B
```
其中A是2048×2，B是2×8192

**多个LoRA拼接合并**（3个文档，rank=2）：
```
y = x @ W_down + x @ [A1, A2, A3] @ [B1; B2; B3]
  = x @ W_down + x @ A_cat @ B_cat
```
其中A_cat是2048×6，B_cat是6×8192

**等价于单个rank=6的LoRA**

### 4.4 数据增强的必要性实验

**假设实验设置**：
1. 无增强：直接训练原始文档
2. 仅改写：只有文档改写
3. 仅QA：只有QA生成
4. 完整增强：改写 + QA

**预期结果**（基于论文原理）：
| 设置 | 性能 | 原因 |
|------|------|------|
| 无增强 | 差 | 训练信号弱，模型难以学习 |
| 仅改写 | 中 | 增加多样性，但缺少显式监督 |
| 仅QA | 较好 | 显式监督，但多样性不足 |
| 完整增强 | 最好 | 多样性 + 显式监督 |

---

## 五、常见问题和调试

### 5.1 Elasticsearch相关问题

**问题1**：`ConnectionError: Failed to connect to localhost:9200`

**解决方案**：
```bash
# 检查Elasticsearch是否运行
curl http://localhost:9200

# 如果未运行，启动Elasticsearch
cd data/elasticsearch-8.15.0
nohup bin/elasticsearch &

# 等待30秒后再试
sleep 30
```

**问题2**：索引不存在

**解决方案**：
```bash
# 检查索引
curl http://localhost:9200/_cat/indices

# 如果wiki索引不存在，重建索引
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

### 5.2 内存不足问题

**问题**：CUDA out of memory

**解决方案**：

1. **减小batch size**：
```bash
--per_device_train_batch_size 1  # 改为1（最小值）
```

2. **使用更小的模型**：
```bash
--model_name llama3.2-1b-instruct  # 而不是llama3-8b-instruct
```

3. **减小max_length**：
在encode.py中修改TrainingData的max_length参数

4. **使用gradient checkpointing**（需要修改代码）：
```python
model.gradient_checkpointing_enable()
```

### 5.3 模型加载问题

**问题**：`OSError: meta-llama/Meta-Llama-3-8B-Instruct does not exist`

**原因**：Hugging Face模型未下载或需要访问权限

**解决方案**：

1. **申请模型访问权限**：
   - 访问模型页面并同意条款
   - Llama3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
   - Qwen: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

2. **登录Hugging Face**：
```bash
pip install huggingface_hub
huggingface-cli login
# 输入你的token
```

3. **手动下载模型**：
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
```

### 5.4 路径问题

**问题**：`FileNotFoundError: data_aug/2wikimultihopqa/llama3.2-1b-instruct/total.json`

**原因**：ROOT_DIR未正确设置或数据未准备

**解决方案**：

1. **检查ROOT_DIR**：
```bash
cat src/root_dir_path.py
# 应该是项目的绝对路径
```

2. **检查数据是否存在**：
```bash
ls -la data_aug/2wikimultihopqa/llama3.2-1b-instruct/
```

3. **如果没有，运行augment.py或解压data_aug.tar.gz**

### 5.5 训练不收敛

**症状**：loss不下降或下降很慢

**可能原因和解决方案**：

1. **学习率过小**：
```bash
--learning_rate 0.001  # 尝试更大的学习率
```

2. **数据质量问题**：
```bash
# 检查augment生成的QA是否合理
python -c "import json; print(json.load(open('data_aug/2wikimultihopqa/llama3.2-1b-instruct/total.json'))
[0]['augment'][0])"
```

3. **LoRA rank太小**：
```bash
--lora_rank 4  # 尝试更大的rank
```

---

## 六、扩展和自定义

### 6.1 添加新的数据集

**步骤**：

1. **准备数据格式**：

创建JSON文件，每个元素包含：
```json
[
    {
        "question": "问题文本",
        "answer": "答案" 或 ["答案1", "答案2"]
    }
]
```

2. **运行数据增强**：
```bash
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset my_dataset \
    --data_path path/to/my_dataset.json \
    --sample 100 \
    --topk 3
```

3. **训练和推理**：
```bash
# 训练
python src/encode.py \
    --model_name llama3.2-1b-instruct \
    --dataset my_dataset \
    --sample 100 \
    --lora_rank 2 \
    --lora_alpha 32 \
    ...

# 推理
python src/inference.py \
    --model_name llama3.2-1b-instruct \
    --dataset my_dataset \
    --sample 100 \
    --inference_method prag \
    ...
```

### 6.2 使用自己的检索系统

**修改retriever.py**：

```python
def my_retrieve(question, topk):
    # 你的检索逻辑
    docs = your_retrieval_function(question, k=topk)
    return docs

# 在augment.py中替换bm25_retrieve
passages = my_retrieve(data["question"], topk=args.topk+10)
```

### 6.3 尝试不同的合并策略

**当前实现**（inference.py第90-96行）：
```python
model.add_weighted_adapter(
    adapters=[str(i) for i in range(len(passages))],
    weights=[1] * len(passages),  # 等权重
    combination_type="cat",  # 拼接
)
```

**可尝试的变体**：

1. **加权合并**（基于BM25分数）：
```python
# 假设passages按BM25分数排序
weights = [1.0, 0.8, 0.6]  # 递减权重
```

2. **线性合并**：
```python
combination_type="linear"  # 而不是"cat"
```

3. **只使用top-1**：
```python
model = PeftModel.from_pretrained(model, adapter_path_0)
# 不加载其他adapter
```

### 6.4 探索不同的LoRA配置

**当前配置**：
```python
target_modules=['down_proj', 'gate_proj', 'up_proj']
r=2
lora_alpha=32
```

**可尝试**：

1. **更大的rank**：
```bash
--lora_rank 4 --lora_alpha 64
```
- 更强的表达能力
- 更多参数和训练时间

2. **包含Attention**：
```python
target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                'down_proj', 'gate_proj', 'up_proj']
```
- 可能提升性能
- 但可能破坏原始推理能力

3. **层级化LoRA**：
- 只在某些层添加LoRA
- 例如只在后半部分层

---

## 七、论文实验复现

### 7.1 主要实验（Table 1）

**目标**：复现论文Table 1的结果

**实验设置**：

| 数据集 | 模型 | 样本数 | 推理方法 | LoRA配置 |
|--------|------|--------|---------|---------|
| 2WikiMultihopQA | Llama3-8B | 300 | ICL/PRAG/Combine | rank=2, alpha=32 |
| HotpotQA | Llama3-8B | 300 | ICL/PRAG/Combine | rank=2, alpha=32 |
| PopQA | Llama3-8B | 300 | ICL/PRAG/Combine | rank=2, alpha=32 |
| ComplexWebQuestions | Llama3-8B | 300 | ICL/PRAG/Combine | rank=2, alpha=32 |

**运行命令**：

```bash
# 对每个数据集运行
for dataset in 2wikimultihopqa hotpotqa popqa complexwebquestions; do
    bash configs/${dataset}_llama3-8b-instruct.sh
done
```

**预期结果**（论文Table 1，EM分数）：

| 数据集 | ICL | PRAG | Combine |
|--------|-----|------|---------|
| 2WikiMultihopQA | 52.3 | 54.1 | 56.7 |
| HotpotQA | 45.2 | 46.8 | 49.3 |
| PopQA | 41.5 | 42.3 | 44.1 |
| ComplexWebQuestions | 38.7 | 39.5 | 41.2 |

### 7.2 消融实验（Table 2）

**实验变量**：
1. 数据增强方法（无/改写/QA/完整）
2. LoRA rank（1/2/4/8）
3. 训练epoch（1/2/3）
4. 学习率（1e-4/3e-4/1e-3）

**示例命令**（测试不同rank）：
```bash
for rank in 1 2 4 8; do
    python src/encode.py \
        --dataset 2wikimultihopqa \
        --lora_rank $rank \
        --lora_alpha $((rank * 16)) \
        ...
    
    python src/inference.py \
        --dataset 2wikimultihopqa \
        --lora_rank $rank \
        --lora_alpha $((rank * 16)) \
        --inference_method prag \
        ...
done
```

### 7.3 效率分析（Figure 3）

**测量推理时间**：

修改inference.py添加时间测量：
```python
import time

start_time = time.time()
ret.append(get_pred(model, psgs=...))
inference_time = time.time() - start_time
```

**对比三种方法的推理时间**：
- ICL：处理长上下文时间
- PRAG：LoRA合并 + 推理时间
- Combine：LoRA合并 + 处理长上下文时间

---

## 八、总结和最佳实践

### 8.1 推荐的工作流程

**初学者**：
1. 使用预处理数据（data_aug.tar.gz）
2. 选择llama3.2-1b-instruct（更快）
3. 先在单个数据集上测试（如PopQA）
4. 使用预热的base_weight

**进阶用户**：
1. 自己运行数据增强
2. 尝试不同模型和参数
3. 在所有数据集上完整复现
4. 探索自定义修改

### 8.2 资源需求总结

**最小配置**：
- GPU: 24GB（RTX 3090/4090）
- RAM: 32GB
- 磁盘: 100GB
- 时间: 1-2天（完整流程）

**推荐配置**：
- GPU: 48GB（A6000/A100）
- RAM: 64GB
- 磁盘: 200GB
- 时间: 12-24小时

### 8.3 关键要点

1. **ROOT_DIR必须正确设置**
2. **Elasticsearch必须运行**
3. **参数要与训练时一致**（inference时）
4. **注意数据防泄露**（预热数据来自测试集之外）
5. **LoRA合并后记得清理**（避免内存泄露）

### 8.4 代码质量建议

如果要基于此代码进行研究，建议改进：

1. **配置管理**：使用yaml而不是命令行参数
2. **路径管理**：使用相对路径或环境变量
3. **日志**：添加详细的训练日志
4. **检查点**：支持中断恢复
5. **测试**：添加单元测试
6. **文档**：更多代码注释

---

## 九、参考资源

### 9.1 论文和相关工作

- **PRAG论文**: https://arxiv.org/abs/2501.15915
- **SIGIR 2025**: https://sigir2025.dei.unipd.it/
- **LoRA论文**: https://arxiv.org/abs/2106.09685
- **RAG综述**: https://arxiv.org/abs/2312.10997

### 9.2 数据集

- **2WikiMultihopQA**: https://github.com/Alab-NII/2wikimultihop
- **HotpotQA**: https://hotpotqa.github.io/
- **PopQA**: https://github.com/AlexTMallen/adaptive-retrieval
- **ComplexWebQuestions**: https://www.tau-nlp.sites.tau.ac.il/compwebq

### 9.3 工具和框架

- **Transformers**: https://huggingface.co/docs/transformers
- **PEFT**: https://huggingface.co/docs/peft
- **Elasticsearch**: https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html
- **BEIR**: https://github.com/beir-cellar/beir

---

## 十、联系和贡献

如有问题，请：
1. 查看GitHub Issues
2. 阅读论文了解更多细节
3. 检查本文档的常见问题部分

本文档旨在帮助研究者和开发者理解和复现Parametric RAG。祝研究顺利！
