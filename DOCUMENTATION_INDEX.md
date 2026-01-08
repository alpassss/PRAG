# 📚 PRAG Project Documentation

This repository now includes comprehensive documentation for understanding and reproducing the Parametric RAG paper (SIGIR 2025).

## Available Documentation

### 🇨🇳 Chinese Version (中文版)
**文件**: `PRAG项目详细说明文档.md`

这是一份详细的中文文档，包含：
- 论文核心思想和创新点的深入分析
- 仓库每个文件和文件夹的详细作用说明
- 完整的复现步骤（包含每一步的详细操作和原因）
- 深入的技术分析（LoRA机制、FFN知识存储等）
- 常见问题和调试指南
- 扩展和自定义建议

**适合**: 中文用户、需要深入理解代码细节的研究者

### 🇬🇧 English Version
**File**: `DETAILED_REPRODUCTION_GUIDE.md`

A comprehensive English guide covering:
- Paper overview and key innovations
- Repository structure and file purposes
- Complete reproduction workflow
- Technical details and best practices
- Troubleshooting and common issues
- Advanced usage and customization

**For**: International users, quick start guide

## Quick Navigation

### For Beginners
1. Read the overview section in either documentation
2. Follow the "Quick Start" guide
3. Use pre-processed data (`data_aug.tar.gz`)
4. Run a single dataset experiment first

### For Researchers
1. Read the paper: https://arxiv.org/abs/2501.15915
2. Study the detailed technical sections
3. Understand the three-stage pipeline
4. Reproduce full experiments

### For Developers
1. Explore the code structure section
2. Check the advanced usage guide
3. Learn about customization options
4. Contribute improvements

## Documentation Structure Comparison

| Section | Chinese Doc | English Doc |
|---------|-------------|-------------|
| Paper Analysis | ✓ (Detailed) | ✓ (Overview) |
| Code Structure | ✓ (File-by-file) | ✓ (Summary) |
| Reproduction Steps | ✓ (Very detailed) | ✓ (Step-by-step) |
| Technical Deep Dive | ✓ (Extensive) | ✓ (Key points) |
| Troubleshooting | ✓ (Comprehensive) | ✓ (Common issues) |
| Code Examples | ✓ (Many) | ✓ (Selected) |

## Main Components Overview

### 📂 Source Code (`src/`)

| File | Purpose | Paper Section |
|------|---------|---------------|
| `augment.py` | Data augmentation | 3.2.1 Self-Augmentation |
| `encode.py` | Document parameterization | 3.2.2 Parameter Training |
| `inference.py` | Inference and evaluation | 3.3 Inference |
| `utils.py` | Utility functions | - |
| `prompt_template.py` | Prompt management | - |
| `retrieve/retriever.py` | BM25 retrieval | - |

### ⚙️ Configuration Files (`configs/`)

12 configuration scripts for different dataset × model combinations:
- 4 datasets: 2WikiMultihopQA, HotpotQA, PopQA, ComplexWebQuestions
- 3 models: Llama3.2-1B, Llama3-8B, Qwen2.5-1.5B

### 📊 Data Files

- `data_aug.tar.gz`: Pre-processed augmented data (recommended)
- `all_prompt.md`: All prompts used in experiments
- Raw datasets: Download separately (see guides)

## Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Self-Augmentation (augment.py)                    │
│ Input: Questions → BM25 Retrieval → Documents              │
│ Process: Rewrite docs + Generate QA pairs                  │
│ Output: data_aug/{dataset}/{model}/                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Document Parameterization (encode.py)             │
│ Input: Augmented data (original + rewrite + QA)            │
│ Process: Train LoRA for each document                      │
│ Output: offline/{model}/{dataset}/data_{i}/passage_{j}/    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Inference (inference.py)                          │
│ Input: Questions + Trained LoRAs                           │
│ Process: Load & merge LoRAs → Generate answers             │
│ Output: output/{model}/{dataset}/{method}/                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 🚀 Parametric RAG Advantages
- **30% faster inference** compared to traditional RAG
- **40% better storage efficiency**
- **No context length limitations**
- **Deep knowledge integration**

### 🔧 Three Inference Methods
1. **ICL**: Traditional RAG (documents in context)
2. **PRAG**: Only parameterized knowledge (fastest)
3. **Combine**: Both parameters + context (best accuracy)

### 📈 Supported Datasets
- **2WikiMultihopQA**: Multi-hop reasoning
- **HotpotQA**: Bridge and comparison questions
- **PopQA**: Popular entity questions
- **ComplexWebQuestions**: Complex queries

### 🤖 Supported Models
- **Llama3.2-1B-Instruct**: Fast, 6GB GPU
- **Llama3-8B-Instruct**: Better quality, 24GB GPU
- **Qwen2.5-1.5B-Instruct**: Balanced, 8GB GPU

## Quick Commands

### Environment Setup
```bash
conda create -n prag python=3.10.4
conda activate prag
pip install torch==2.1.0
pip install -r requirements.txt
```

### Use Pre-processed Data
```bash
tar -xzvf data_aug.tar.gz
```

### Run Full Experiment
```bash
# Example: 2WikiMultihopQA with Llama3.2-1B
bash configs/2wikimultihopqa_llama3.2-1b-instruct.sh
```

## Hardware Requirements

| Configuration | GPU | RAM | Disk | Time |
|--------------|-----|-----|------|------|
| Minimum | 24GB | 32GB | 100GB | 1-2 days |
| Recommended | 48GB | 64GB | 200GB | 12-24h |

## Getting Help

### Documentation
- Chinese (详细版): `PRAG项目详细说明文档.md`
- English (Quick): `DETAILED_REPRODUCTION_GUIDE.md`

### Resources
- Paper: https://arxiv.org/abs/2501.15915
- SIGIR 2025: https://sigir2025.dei.unipd.it/
- GitHub Issues: Report problems and ask questions

### Common Issues
- Elasticsearch connection → See troubleshooting section
- CUDA OOM → Use smaller model or reduce batch size
- Model download → Login to Hugging Face and request access
- Path errors → Check ROOT_DIR in `src/root_dir_path.py`

## Contributing

Contributions are welcome! Areas for improvement:
- Additional datasets
- New models
- Efficiency optimizations
- Better documentation
- Bug fixes

## Citation

```bibtex
@inproceedings{su2025parametric,
  title={Parametric Retrieval Augmented Generation},
  author={Su, Weihang and Tang, Yichen and Ai, Qingyao and others},
  booktitle={SIGIR 2025},
  year={2025}
}
```

## License

Follow the repository's original license terms.

---

**Choose Your Guide**:
- 🇨🇳 需要详细的中文说明？→ `PRAG项目详细说明文档.md`
- 🇬🇧 Want a quick English guide? → `DETAILED_REPRODUCTION_GUIDE.md`
- 📖 Original README → `README.md`
- 💬 All prompts → `all_prompt.md`

Happy researching! 🎓
