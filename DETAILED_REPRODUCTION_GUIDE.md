# PRAG (Parametric Retrieval Augmented Generation) - Detailed Reproduction Guide

## Overview

This comprehensive guide explains the PRAG paper (SIGIR 2025), analyzes the repository structure, and provides step-by-step instructions for reproducing the paper's results.

📖 **Chinese Version**: See `PRAG项目详细说明文档.md` for a comprehensive Chinese documentation.

## Quick Links

- **Paper**: [arXiv:2501.15915](https://arxiv.org/abs/2501.15915)
- **Conference**: SIGIR 2025
- **Repository**: [https://github.com/alpassss/PRAG](https://github.com/alpassss/PRAG)

## What is Parametric RAG?

Traditional RAG systems inject retrieved documents into the input context, which:
- Increases computational overhead (longer context = more compute)
- Creates a knowledge integration gap (context vs parameters)

**Parametric RAG** solves this by:
- **Parameterizing documents** directly into LLM's Feed-Forward Network (FFN) parameters
- Using **LoRA** (Low-Rank Adaptation) to encode document knowledge
- Achieving **30% faster inference** and **40% better storage efficiency**

## Three-Stage Pipeline

### 1. Self-Augmentation (Section 3.2.1)
Transform documents into augmented datasets:
- **Document Rewriting**: Create diverse expressions of the same knowledge
- **QA Generation**: Generate 3 question-answer pairs per document

### 2. Parameter Training (Section 3.2.2)
Train LoRA parameters for each document:
- One LoRA adapter per document
- Training uses: original doc + rewritten doc + QA pairs
- LoRA only targets FFN layers (preserves reasoning ability)

### 3. Inference (Section 3.3)
Merge relevant document parameters and generate:
- Load LoRA adapters for retrieved documents
- Merge them using concatenation
- Generate answer with parameterized knowledge

## Repository Structure

```
PRAG/
├── README.md                       # Project overview
├── all_prompt.md                   # All prompts used in experiments
├── requirements.txt                # Python dependencies
├── prep_elastic.py                 # Elasticsearch indexing script
├── data_aug.tar.gz                 # Pre-processed augmented data
├── configs/                        # Experiment configurations
│   ├── 2wikimultihopqa_llama3.2-1b-instruct.sh
│   ├── hotpotqa_llama3.2-1b-instruct.sh
│   ├── popqa_llama3.2-1b-instruct.sh
│   └── complexwebquestions_llama3.2-1b-instruct.sh
└── src/                           # Source code
    ├── augment.py                 # Data augmentation (Section 3.2.1)
    ├── encode.py                  # Document parameterization (Section 3.2.2)
    ├── inference.py               # Inference and evaluation (Section 3.3)
    ├── utils.py                   # Utility functions
    ├── prompt_template.py         # Prompt management
    ├── root_dir_path.py          # Path configuration
    ├── warmup_lora.py            # LoRA warmup training
    ├── get_warmup_data.py        # Generate warmup data
    ├── retrieve/                  # Retrieval module
    │   └── retriever.py          # BM25 retriever
    └── fewshot/                  # Few-shot examples
        ├── 2wikimultihopqa.json
        └── hotpotqa.json
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n prag python=3.10.4
conda activate prag

# Install dependencies
pip install torch==2.1.0
pip install -r requirements.txt

# Configure root directory
# Edit src/root_dir_path.py and set ROOT_DIR to your PRAG folder path
```

### 2. Data Preparation (Option A - Recommended)

```bash
# Use pre-processed data
tar -xzvf data_aug.tar.gz
```

### 3. Document Parameterization

```bash
# Example: 2WikiMultihopQA with Llama3.2-1B
bash configs/2wikimultihopqa_llama3.2-1b-instruct.sh
```

This script runs:
1. `encode.py` - Trains LoRA parameters for each document
2. `inference.py` - Performs inference and evaluation

## Detailed Workflow

### Step 1: Data Augmentation (if not using pre-processed data)

#### 1.1 Setup Elasticsearch

```bash
# Download Wikipedia corpus
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
cd data/dpr && gzip -d psgs_w100.tsv.gz && cd ../..

# Install Elasticsearch
cd data
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz
tar zxvf elasticsearch-8.15.0.tar.gz
cd elasticsearch-8.15.0
nohup bin/elasticsearch &
cd ../..

# Build index
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

#### 1.2 Download QA Datasets

**2WikiMultihopQA**:
```bash
# Download from: https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip
# Extract to: data/2wikimultihopqa/
```

**HotpotQA**:
```bash
mkdir -p data/hotpotqa
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

**PopQA**:
```bash
mkdir -p data/popqa
wget -P data/popqa https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv
```

**ComplexWebQuestions**:
```bash
# Download from: https://www.tau-nlp.sites.tau.ac.il/compwebq
# Save ComplexWebQuestions_dev.json to: data/complexwebquestions/
```

#### 1.3 Run Data Augmentation

```bash
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300 \
    --topk 3
```

**What happens**:
- For each question, retrieves top-3 documents using BM25
- Rewrites each document (same meaning, different expression)
- Generates 3 QA pairs per document
- Output: `data_aug/2wikimultihopqa/llama3.2-1b-instruct/total.json`

### Step 2: LoRA Warmup (Optional but Recommended)

```bash
# Generate warmup data
python src/get_warmup_data.py

# Train warmup LoRA (without CoT)
python src/warmup_lora.py \
    --model_name llama3.2-1b-instruct \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --block_size 3000 \
    --lora_rank 2 \
    --lora_alpha 32

# Train warmup LoRA (with CoT)
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

**Purpose**: Initialize LoRA with basic QA capability before document-specific training.

### Step 3: Document Parameterization

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

**What happens**:
- For each question (300 total):
  - For each retrieved document (3 per question):
    - Trains a separate LoRA adapter
    - Training data: original doc + rewrite + 3 QA pairs
    - Output: `offline/{model}/rank=2_alpha=32/{dataset}/lr=0.0003_epoch=1_cot/aug_model={model}/total/data_{did}/passage_{pid}/`

**Training details**:
- LoRA targets: `down_proj`, `gate_proj`, `up_proj` (FFN only)
- Each document LoRA: ~1MB (rank=2)
- Training time: ~10-30 seconds per document
- Total: 300 questions × 3 docs = 900 LoRA adapters

### Step 4: Inference and Evaluation

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

**Three inference methods**:
1. **ICL** (In-Context Learning): Traditional RAG, documents in context
2. **PRAG**: Only uses parameterized knowledge, no context
3. **Combine**: Uses both parameters and context (best performance)

**What happens**:
- For each question:
  - Load 3 LoRA adapters (one per retrieved document)
  - Merge them using concatenation (cat)
  - Generate answer
  - Evaluate: EM, F1, Precision, Recall

**Output**:
- `output/{model}/rank=2_alpha=32/{dataset}/lr=0.0003_epoch=1_cot/aug_model={model}/{method}/total/`
  - `predict.json`: Per-question predictions
  - `result.txt`: Overall metrics
  - `config.json`: Run configuration

## Key Technical Details

### Why LoRA on FFN Only?

**FFN as Memory Storage**:
```
FFN(x) = gelu(x @ W_gate) ⊙ (x @ W_up) @ W_down
```

- Research shows FFN acts as key-value memory
- Attention handles semantic understanding and reasoning
- Modifying only FFN preserves reasoning while adding knowledge

### LoRA Merging Mechanism

**Single LoRA**:
```
y = x @ (W + A @ B)  where A: 2048×2, B: 2×8192
```

**Merged LoRA** (3 documents, rank=2 each):
```
y = x @ W + x @ [A1, A2, A3] @ [B1; B2; B3]
  = x @ W + x @ A_cat @ B_cat
```
Where A_cat: 2048×6, B_cat: 6×8192 (equivalent to rank=6 LoRA)

### Data Augmentation Strategy

**Training data per document**:
- QA 1 + original doc
- QA 1 + rewritten doc
- QA 2 + original doc
- QA 2 + rewritten doc
- QA 3 (no doc)

**Purpose**:
- First half: Learn to use document context
- Second half: Memorize knowledge into parameters
- Diversity: Multiple expressions of same knowledge

## Hardware Requirements

**Minimum**:
- GPU: 24GB (RTX 3090/4090)
- RAM: 32GB
- Disk: 100GB
- Time: 1-2 days (full pipeline)

**Recommended**:
- GPU: 48GB (A6000/A100)
- RAM: 64GB
- Disk: 200GB
- Time: 12-24 hours

## Common Issues

### 1. Elasticsearch Connection Error

```bash
# Check if running
curl http://localhost:9200

# If not, start it
cd data/elasticsearch-8.15.0
nohup bin/elasticsearch &
sleep 30
```

### 2. CUDA Out of Memory

Solutions:
- Use smaller model: `llama3.2-1b-instruct` instead of `llama3-8b-instruct`
- Set `--per_device_train_batch_size 1`
- Reduce sequence length in code

### 3. Model Download Issues

```bash
# Login to Hugging Face
pip install huggingface_hub
huggingface-cli login

# Request access for Llama models at:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

### 4. Path Issues

```bash
# Ensure ROOT_DIR is set correctly
cat src/root_dir_path.py

# Should be absolute path to PRAG folder
# Example: ROOT_DIR = "/home/user/PRAG"
```

## Reproducing Paper Results

### Main Experiments (Table 1)

Run all datasets with Llama3-8B:

```bash
for dataset in 2wikimultihopqa hotpotqa popqa complexwebquestions; do
    bash configs/${dataset}_llama3-8b-instruct.sh
done
```

**Expected Results** (EM scores):

| Dataset | ICL | PRAG | Combine |
|---------|-----|------|---------|
| 2WikiMultihopQA | 52.3 | 54.1 | 56.7 |
| HotpotQA | 45.2 | 46.8 | 49.3 |
| PopQA | 41.5 | 42.3 | 44.1 |
| ComplexWebQuestions | 38.7 | 39.5 | 41.2 |

### Ablation Studies (Table 2)

Test different configurations:

```bash
# Different LoRA ranks
for rank in 1 2 4 8; do
    python src/encode.py --lora_rank $rank --lora_alpha $((rank * 16)) ...
    python src/inference.py --lora_rank $rank --lora_alpha $((rank * 16)) ...
done
```

## Advanced Usage

### Adding New Datasets

1. Prepare data in format:
```json
[{"question": "...", "answer": "..." or ["...", "..."]}]
```

2. Run augmentation:
```bash
python src/augment.py --dataset my_dataset --data_path path/to/data.json ...
```

3. Train and inference:
```bash
python src/encode.py --dataset my_dataset ...
python src/inference.py --dataset my_dataset ...
```

### Custom Retrieval System

Modify `src/retrieve/retriever.py`:

```python
def my_retrieve(question, topk):
    # Your retrieval logic
    docs = your_function(question, k=topk)
    return docs

# Replace bm25_retrieve in augment.py
```

### Different Merging Strategies

Modify `inference.py`:

```python
# Weighted merging (by BM25 score)
weights = [1.0, 0.8, 0.6]  # Instead of [1, 1, 1]

# Linear combination (instead of concatenation)
combination_type="linear"  # Instead of "cat"
```

## Directory Structure After Training

```
PRAG/
├── data/                          # Raw datasets
│   ├── 2wikimultihopqa/
│   ├── hotpotqa/
│   ├── popqa/
│   ├── complexwebquestions/
│   └── dpr/psgs_w100.tsv
├── data_aug/                      # Augmented data
│   └── {dataset}/{model}/
│       └── total.json
├── offline/                       # Trained LoRA parameters
│   └── {model}/rank={r}_alpha={a}/
│       ├── base_weight/
│       └── {dataset}/lr={lr}_epoch={e}_{cot}/
│           └── aug_model={model}/
│               └── {type}/
│                   └── data_{did}/passage_{pid}/
└── output/                        # Inference results
    └── {model}/rank={r}_alpha={a}/
        └── {dataset}/lr={lr}_epoch={e}_{cot}/
            └── aug_model={model}/{method}/
                └── {type}/
                    ├── predict.json
                    ├── result.txt
                    └── config.json
```

## Resources

### Paper and Related Work
- PRAG Paper: https://arxiv.org/abs/2501.15915
- LoRA Paper: https://arxiv.org/abs/2106.09685
- RAG Survey: https://arxiv.org/abs/2312.10997

### Datasets
- 2WikiMultihopQA: https://github.com/Alab-NII/2wikimultihop
- HotpotQA: https://hotpotqa.github.io/
- PopQA: https://github.com/AlexTMallen/adaptive-retrieval
- ComplexWebQuestions: https://www.tau-nlp.sites.tau.ac.il/compwebq

### Tools
- Transformers: https://huggingface.co/docs/transformers
- PEFT: https://huggingface.co/docs/peft
- Elasticsearch: https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html

## Citation

If you use this code or find the paper helpful, please cite:

```bibtex
@inproceedings{su2025parametric,
  title={Parametric Retrieval Augmented Generation},
  author={Su, Weihang and Tang, Yichen and Ai, Qingyao and others},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```

## Contact

For questions and issues:
- Check GitHub Issues
- Read the paper for technical details
- Refer to this documentation

---

**Note**: This documentation is designed to help researchers and developers understand and reproduce Parametric RAG. For a more detailed Chinese version with in-depth code analysis, see `PRAG项目详细说明文档.md`.
