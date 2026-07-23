---
layout: default
title: Training
nav_order: 4
has_children: true
permalink: /training/
---

# Training Emotion-LLaMA
{: .no_toc }

Complete guide to training your own Emotion-LLaMA model from scratch.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Training Overview

Emotion-LLaMA training consists of two stages:

1. **Stage 1: Pre-training** - Train on coarse-grained MERR dataset (28,618 samples)
2. **Stage 2: Instruction Tuning** - Fine-tune on fine-grained MERR dataset (4,487 samples)

This two-stage approach enables the model to:
- Learn basic multimodal emotion recognition in Stage 1
- Develop advanced emotion reasoning capabilities in Stage 2

---

## Prerequisites

### Hardware Requirements

- **GPUs**: 4x NVIDIA GPUs with at least 24GB VRAM each (e.g., RTX 3090, A5000, or better)
- **RAM**: 64GB or more recommended
- **Storage**: 100GB+ free space for datasets and checkpoints

### Software Requirements

- ✅ Emotion-LLaMA environment installed (see [Getting Started](../getting-started/))
- ✅ PyTorch with CUDA support
- ✅ Distributed training libraries (automatically included in environment)

---

## Stage 1: Pre-training

### Step 1: Download the Dataset

{: .warning }
> Due to copyright restrictions, we cannot provide raw videos directly.

Visit the official MER2023 website to apply for dataset access:
```
http://merchallenge.cn/datasets
```

After obtaining access, configure the dataset paths in the Stage 1 training
YAML as shown in Step 3.

### Step 2: Prepare Multi-modal Encoders

To extract rich emotion features, we use:
- **HuBERT** - Audio Encoder
- **EVA** - Global Visual Encoder
- **MAE** - Local Visual Encoder
- **VideoMAE** - Temporal Encoder

{: .tip }
> To save GPU memory, we use pre-extracted features instead of loading all encoders directly.

Download the pre-extracted features:
[Google Drive Link](https://drive.google.com/drive/folders/1DqGSBgpRo7TuGNqMJo9BYg6smJE20MG4?usp=drive_link)

Save the features to your dataset folder and configure the three feature roots
in YAML. FaceMAE, VideoMAE, and HuBERT feature files are all required; the
training dataset does not extract them online.

{: .note }
> The specific feature extraction process is available in the "feature_extract" folder: [Google Drive Link](https://drive.google.com/drive/folders/1d-Sg5fAskt2s6OOEUNXFaM2u-C055Whj?usp=sharing)

### Step 3: Configure Dataset

Set the following keys in `train_configs/Emotion-LLaMA_finetune.yaml` (or
override the same keys in the default dataset YAML):

```yaml
datasets:
  feature_face_caption:
    task_pool: [emotion, reason]
    annotation_format: auto
    build_info:
      image_path: /path/to/MER2023/video
      ann_path: /path/to/MER2023/MERR_coarse_grained.txt
      transcription_path: transcription_en_all.csv
      coarse_grained_json_path: MERR_coarse_grained.json
      face_feature_path: mae_340_UTT
      video_feature_path: maeV_399_UTT
      audio_feature_path: HL-UTT
```

Relative metadata and feature paths are resolved from the directory containing
`ann_path`. The transcription CSV is optional but recommended. This setup uses
28,618 coarse-grained samples for pre-training.

### Step 4: Configure Multi-task Instructions

`task_pool: [emotion, reason]` in the YAML above enables multimodal emotion
recognition and coarse-grained emotion inference. `reason_v2` is reserved for
the Stage 2 fine-grained configuration.

Each task randomly selects prompts from different instruction pools:

**Emotion Task Examples:**
- "What is the emotion expressed in this video?"
- "Identify the primary emotion shown."
- "Classify the emotional state."

**Reason Task Examples:**
- "What are the facial expressions and vocal tone used? What emotion does this reflect?"
- "Analyze the multimodal cues and explain the emotion."
- "Why is this person experiencing this emotion?"

### Step 5: Run Pre-training

Execute the training script with 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 train.py --cfg-path train_configs/Emotion-LLaMA_finetune.yaml
```

**Training Configuration** (`train_configs/Emotion-LLaMA_finetune.yaml`):
```yaml
model:
  arch: minigpt_v2
  llama_model: "/path/to/checkpoints/Llama-2-7b-chat-hf"
  ckpt: "/path/to/checkpoints/minigptv2_checkpoint.pth"
  lora_r: 64
  lora_alpha: 16

datasets:
  feature_face_caption:
    batch_size: 1

run:
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-5
  min_lr: 1e-6
  warmup_lr: 1e-6
  weight_decay: 0.05
  max_epoch: 30
  num_workers: 6
  iters_per_epoch: 1000
  warmup_steps: 1000
  seed: 42
  amp: True
```

### Training Progress

Monitor the training with:
- **Training loss**: Should decrease steadily
- **Validation metrics**: Available only after explicit validation is enabled
- **Checkpoints**: Saved in `checkpoints/save_checkpoint/`

Expected training time: ~2-3 days on 4x A5000 GPUs

---

## Stage 2: Instruction Tuning

For advanced emotion reasoning with fine-grained annotations, see [Instruction Tuning](instruction-tuning.md).

---

## Training Tips

### Optimize GPU Memory

If you encounter out-of-memory errors:

1. **Batch size is already 1** (per GPU)
   - With 4 GPUs, effective batch size = 4

2. **Mixed precision is enabled by default**:
   ```yaml
   amp: True  # Already enabled
   ```

3. **Use gradient checkpointing**:
   ```yaml
   use_grad_checkpoint: True  # Already enabled
   ```

4. **Reduce image size** (if desperate):
   ```yaml
   image_size: 224  # Instead of 448
   ```

### Monitor Training

Use TensorBoard to visualize training:

```bash
tensorboard --logdir=checkpoints/save_checkpoint/
```

View metrics at: `http://localhost:6006`

### Resume Training

If training is interrupted, resume from the last checkpoint:

```yaml
resume_ckpt_path: "checkpoints/save_checkpoint/checkpoint_15.pth"
```

Runs with validation also maintain `checkpoint_last.pth`; resume from that file
to preserve the best-metric and early-stopping state. Keep its sibling
`checkpoint_best.pth` as well: a resumed run can reuse that historical best
even though the new job writes to a different output directory.

---

## Evaluation During Training

Validation is disabled by default. It is enabled only when both the dataset
and the runner explicitly name a validation split. The legacy
`build_info.ann_path` form still creates a training split only, so existing
training configurations do not start reading held-out data unexpectedly.

First configure deterministic evaluation processors, one evaluation task, and
explicit annotation files. `annotations` takes precedence over the legacy
`ann_path` key:

```yaml
datasets:
  feature_face_caption:
    batch_size: 1
    task_pool: [emotion, reason]
    evaluation_task: emotion
    labels: [neutral, angry, happy, sad, worried, surprise]
    vis_processor:
      train:
        name: blip2_image_train
        image_size: 448
      eval:
        name: blip2_image_eval
        image_size: 448
    text_processor:
      train:
        name: blip_caption
      eval:
        name: blip_caption
    build_info:
      image_path: /path/to/MER2023/video
      annotations:
        train: /path/to/MER2023/train.txt
        val: /path/to/MER2023/val.txt
        # test: /path/to/MER2023/test.txt
      transcription_path: transcription_en_all.csv
      coarse_grained_json_path: MERR_coarse_grained.json
      face_feature_path: mae_340_UTT
      video_feature_path: maeV_399_UTT
      audio_feature_path: HL-UTT
```

Then opt in from the runner:

```yaml
run:
  evaluate: false
  train_splits: [train]
  valid_splits: [val]
  test_splits: []

  metric_for_best_model: macro_f1
  best_model_split: val
  greater_is_better: true
  early_stopping_patience: 5  # null disables early stopping
  early_stopping_min_delta: 0.0

  evaluation:
    task: classification
    labels: [neutral, angry, happy, sad, worried, surprise]
    generation:
      max_new_tokens: 20
      num_beams: 1
      do_sample: false
```

Validation runs at the end of every training epoch. It writes structured
artifacts under `result/<split>/epoch_<epoch>/`, saves improvements to
`checkpoint_best.pth`, and refreshes `checkpoint_last.pth` after every
validated epoch. The checkpoint stores best metric, best epoch, and patience
state, so resuming does not reset early stopping.

`valid_splits` and `test_splits` must be disjoint and must exist in
`build_info.annotations`. A `test` annotation is never inferred or reused as
validation data. Multiple validation split names are supported for reporting;
`best_model_split` selects the one used for checkpoint decisions. Each
validation or test split currently accepts one dataset source.

{: .warning }
> `evaluate: true` means **evaluation-only**: training is skipped and only the
> explicitly configured `test_splits` are evaluated. It is not the switch for
> training-time validation. In evaluation-only mode, set `train_splits: []`,
> `valid_splits: []`, and provide at least one `test_splits` entry.

---

## Hyperparameter Tuning

Key hyperparameters to tune:

| Parameter | Description | Default | Range |
|:----------|:------------|:-------:|:------|
| `init_lr` | Initial learning rate | 1e-5 | 1e-6 to 1e-4 |
| `batch_size` | Batch size per GPU | 1 | 1 to 2 |
| `max_epoch` | Number of epochs | 30 | 20 to 50 |
| `warmup_steps` | Warmup steps | 1000 | 500 to 2000 |
| `weight_decay` | Weight decay | 0.05 | 0.01 to 0.1 |
| `lora_r` | LoRA rank | 64 | 32 to 128 |
| `lora_alpha` | LoRA alpha | 16 | 8 to 32 |

---

## Troubleshooting

### Common Issues

**Issue**: "CUDA out of memory"
- **Solution**: Reduce batch size or enable gradient accumulation

**Issue**: "NaN loss during training"
- **Solution**: Reduce learning rate or enable gradient clipping

**Issue**: "Slow training speed"
- **Solution**: Increase `num_workers` or use pre-extracted features

**Issue**: "Model not converging"
- **Solution**: Check data preprocessing and try different learning rates

---

## Next Steps

After completing Stage 1 pre-training:

1. [Continue to Instruction Tuning (Stage 2)](instruction-tuning.md)
2. [Evaluate your trained model](../evaluation/)
3. [Deploy the model for inference](../demo/)

---

## Questions?

For training-related questions:
- Check the [Troubleshooting section](#troubleshooting)
- Open an [issue on GitHub](https://github.com/ZebangCheng/Emotion-LLaMA/issues)
- Review the original [training configuration files](https://github.com/ZebangCheng/Emotion-LLaMA/tree/main/train_configs)

