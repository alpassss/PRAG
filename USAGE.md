# Parametric RAG - Quick Start Guide

This guide provides step-by-step instructions to run the Parametric RAG experiments.

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- At least 16GB GPU memory for llama3-8b-instruct

## Step 1: Install Environment

```bash
# Create and activate conda environment
conda create -n prag python=3.10
conda activate prag

# Install PyTorch (adjust for your CUDA version)
pip install torch>=2.0.0

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Data

Extract the pre-augmented data:

```bash
tar -xzvf data_aug.tar.gz
```

This creates the `data_aug/` directory with augmented data for each dataset.

## Step 3: Train Document Adapters (Encode Step)

Train LoRA adapters for each passage in the dataset. Each passage gets its own adapter that encodes its knowledge.

### Example: HotpotQA with LLaMA-3-8B

```bash
python src/encode.py \
    --model_name llama3-8b-instruct \
    --dataset hotpotqa \
    --sample 300 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot
```

### Example: PopQA (without CoT)

```bash
python src/encode.py \
    --model_name llama3-8b-instruct \
    --dataset popqa \
    --sample 300 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 2 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32
```

The trained adapters are saved in the `offline/` directory.

## Step 4: Run Inference

There are three inference methods:

### 1. ICL Mode (Traditional RAG)

- **Description**: Uses retrieved passages in the prompt, no adapters
- **How it works**: Base model + passages in context

```bash
python src/inference.py \
    --model_name llama3-8b-instruct \
    --dataset hotpotqa \
    --sample 300 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --max_new_tokens 128 \
    --inference_method icl \
    --with_cot
```

### 2. PRAG Mode (Parametric RAG)

- **Description**: Merges trained adapters into model, no passages in prompt
- **How it works**: Adapters contain encoded knowledge → merged into model weights

```bash
python src/inference.py \
    --model_name llama3-8b-instruct \
    --dataset hotpotqa \
    --sample 300 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --max_new_tokens 128 \
    --inference_method prag \
    --with_cot
```

### 3. Combine Mode

- **Description**: Uses both adapters AND passages in prompt
- **How it works**: Merged adapters + passages in context

```bash
python src/inference.py \
    --model_name llama3-8b-instruct \
    --dataset hotpotqa \
    --sample 300 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --max_new_tokens 128 \
    --inference_method combine \
    --with_cot
```

## Expected Output Structure

```
output/
├── {model_name}/
│   └── rank={lora_rank}_alpha={lora_alpha}/
│       └── {dataset}/
│           └── lr={learning_rate}_epoch={num_train_epochs}_{cot|direct}/
│               └── aug_model={model_name}/
│                   └── {icl|prag|combine}/
│                       └── {data_type}/
│                           ├── config.json
│                           ├── predict.json
│                           └── result.txt
```

## Supported Models

| Argument | HuggingFace Model |
|----------|-------------------|
| `llama3-8b-instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `llama3.2-1b-instruct` | `meta-llama/Llama-3.2-1B-Instruct` |
| `qwen2.5-1.5b-instruct` | `Qwen/Qwen2.5-1.5B-Instruct` |

## Supported Datasets

- `hotpotqa`
- `2wikimultihopqa`
- `popqa`
- `complexwebquestions`

## Key Differences Between Modes

| Mode | Adapters Merged? | Passages in Prompt? | Use Case |
|------|-----------------|---------------------|----------|
| ICL | No | Yes | Baseline RAG |
| PRAG | Yes | No | Knowledge in parameters |
| Combine | Yes | Yes | Best of both worlds |

## Troubleshooting

### Out of Memory

- Reduce `--sample` to process fewer examples
- Use a smaller model like `llama3.2-1b-instruct`
- Reduce `--max_new_tokens`

### Adapter Not Found

Make sure you ran the encode step first with matching parameters:
- Same `--model_name`
- Same `--dataset`
- Same `--lora_rank` and `--lora_alpha`
- Same `--num_train_epochs` and `--learning_rate`
- Same `--with_cot` flag

