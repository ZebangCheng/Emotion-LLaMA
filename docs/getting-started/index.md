---
layout: default
title: Getting Started
nav_order: 2
has_children: true
permalink: /getting-started/
---

# Getting Started
{: .no_toc }

This guide will help you set up Emotion-LLaMA on your system and get started with multimodal emotion recognition.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (for training)
- **RAM**: 32GB or more recommended
- **Storage**: At least 50GB free space for models and datasets

### Software Requirements

- **Operating System**: Linux (Ubuntu 18.04+), Windows 10/11, or macOS
- **Python**: 3.8 or later
- **CUDA**: 11.0 or later (for GPU acceleration)
- **Conda**: Anaconda or Miniconda

---

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ZebangCheng/Emotion-LLaMA.git
cd Emotion-LLaMA
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate llama
```

{: .note }
> The environment setup may take several minutes depending on your internet connection.

### 3. Install Additional Dependencies

```bash
pip install moviepy==1.0.3
pip install soundfile==0.12.1
pip install opencv-python==4.7.0.72
```

---

## Preparing Pre-trained Models

### Llama-2-7b-chat-hf

Download the Llama-2-7b-chat-hf model from Hugging Face:

```
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

Save the model to `Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf/`

### Configure Model Path

Specify the path to Llama-2 in the model config file (`minigpt4/configs/models/minigpt_v2.yaml`):

```yaml
# Set Llama-2-7b-chat-hf path
llama_model: "/path/to/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf"
```

### MiniGPT-v2 Checkpoint

Specify the path to MiniGPT-v2 in the training config file (`train_configs/Emotion-LLaMA_finetune.yaml`):

```yaml
# Set MiniGPT-v2 path
ckpt: "/path/to/Emotion-LLaMA/checkpoints/minigptv2_checkpoint.pth"
```

{: .warning }
> Replace `/path/to/` with your actual installation path.

### HuBERT-large Model (For Demo)

Download the HuBERT-large model from Hugging Face:

```
https://huggingface.co/TencentGameMate/chinese-hubert-large
```

Save to `Emotion-LLaMA/checkpoints/transformer/chinese-hubert-large/`

Specify the path in `eval_configs/demo.yaml`:

```yaml
# Set HuBERT-large model path
model:
  audio_model_path: "checkpoints/transformer/chinese-hubert-large"
```

---

## Project Structure

The complete project structure including datasets and checkpoints:

```
📦 Dataset (External)
 └── 📦 Emotion
     └── 📂 MER2023
         ├── 📂 video                      # Raw video files
         ├── 📂 HL-UTT                     # HuBERT features
         ├── 📂 mae_340_UTT                # MAE features
         ├── 📂 maeV_399_UTT               # VideoMAE features
         ├── 📄 transcription_en_all.csv   # Video transcriptions
         ├── 📄 MERR_coarse_grained.txt    # 28,618 coarse-grained annotations
         ├── 📄 MERR_coarse_grained.json
         ├── 📄 MERR_fine_grained.txt      # 4,487 fine-grained annotations
         └── 📄 MERR_fine_grained.json

📦 Emotion-LLaMA (Project Root)
 ├── 📂 checkpoints/                      # Pre-trained models
 │   ├── 📂 Llama-2-7b-chat-hf/          # Base LLaMA model
 │   ├── 📂 save_checkpoint/             # Trained checkpoints
 │   │   ├── 📂 stage2/
 │   │   │   ├── checkpoint_best.pth     # Best model from stage 2
 │   │   │   └── log.txt                 # Training logs
 │   │   └── Emoation_LLaMA.pth          # Demo model
 │   ├── 📂 transformer/
 │   │   └── 📂 chinese-hubert-large/    # Audio encoder
 │   └── minigptv2_checkpoint.pth        # MiniGPT-v2 base
 ├── 📂 minigpt4/                        # Core model implementation
 │   ├── 📂 configs/                     # Model and dataset configs
 │   ├── 📂 datasets/                    # Dataset loaders
 │   ├── 📂 models/                      # Model architectures
 │   ├── 📂 processors/                  # Data processors
 │   └── 📂 conversation/                # Conversation templates
 ├── 📂 train_configs/                   # Training configurations
 │   ├── Emotion-LLaMA_finetune.yaml     # Stage 1 config
 │   └── minigptv2_tuning_stage_2.yaml   # Stage 2 config
 ├── 📂 eval_configs/                    # Evaluation configurations
 │   ├── demo.yaml                       # Demo config
 │   ├── eval_emotion.yaml               # MER2023 eval config
 │   └── eval_emotion_EMER.yaml          # EMER eval config
 ├── 📂 examples/                        # Example video clips
 ├── 📂 images/                          # Documentation images
 ├── 📂 docs/                            # Documentation site (this site!)
 ├── 📑 train.py                         # Training script
 ├── 📑 eval_emotion.py                  # Evaluation script (MER2023)
 ├── 📑 eval_emotion_EMER.py             # Evaluation script (EMER)
 ├── 📑 app.py                           # Gradio demo
 ├── 📑 app_EmotionLlamaClient.py        # API client
 ├── 📜 environment.yml                  # Conda environment
 └── 📜 requirements.txt                 # Python dependencies
```

### Key Directories

**Checkpoints**: Store all pre-trained models and trained checkpoints
- Download Llama-2-7b-chat-hf from Hugging Face
- Save trained models from training runs

**Dataset**: External dataset directory (apply for access)
- Must be organized as shown above
- Features should be pre-extracted for faster training

**Documentation**: This documentation site
- Built with Jekyll and Just the Docs theme
- Deployed to GitHub Pages

---

## Verification

To verify your installation, you can run a quick test:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
```

---

## Next Steps

- [Try the Demo](../demo/) - Run Emotion-LLaMA online or locally
- [Explore the Dataset](../dataset/) - Learn about the MERR dataset
- [Train Your Model](../training/) - Train Emotion-LLaMA from scratch
- [Run Evaluations](../evaluation/) - Evaluate model performance

---

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory error
- **Solution**: Reduce batch size in configuration files or use gradient accumulation

**Issue**: Missing dependencies
- **Solution**: Reinstall the conda environment: `conda env remove -n llama && conda env create -f environment.yaml`

**Issue**: Model download fails
- **Solution**: Check your internet connection and Hugging Face access permissions

For more help, please [open an issue](https://github.com/ZebangCheng/Emotion-LLaMA/issues) on GitHub.

