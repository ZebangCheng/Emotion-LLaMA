---
layout: default
title: Instruction Tuning
parent: Training
nav_order: 1
---

# Stage 2: Instruction Tuning
{: .no_toc }

Advanced training stage for enhanced emotion reasoning capabilities.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Stage 2 (Instruction Tuning) focuses on enhancing Emotion-LLaMA's ability to:
- Reason about emotions using multimodal cues
- Generate detailed explanations of emotional states
- Understand complex emotional contexts
- Provide fine-grained emotion analysis

This stage uses the **4,487 fine-grained annotated samples** from the MERR dataset, which include comprehensive multimodal descriptions and reasoning.

---

## Prerequisites

Before starting Stage 2, you should have:

✅ Completed [Stage 1 Pre-training](index.md)  
✅ A trained checkpoint from Stage 1  
✅ Fine-grained MERR annotations downloaded  
✅ Same environment and dataset setup as Stage 1

---

## Configuration

### Step 1: Set Pre-trained Model Path

If you completed Stage 1 pre-training, use the final checkpoint:

**In `train_configs/minigptv2_tuning_stage_2.yaml`:**
```yaml
# Set Emotion-LLaMA path from Stage 1
ckpt: "/path/to/Emotion-LLaMA/checkpoints/save_checkpoint/2024xxxx-v1/checkpoint_29.pth"
```

{: .tip }
> Use the checkpoint from the last epoch of Stage 1 for best results.

**Alternative**: If you didn't run Stage 1, use the provided demo model:

```yaml
# Set Emotion-LLaMA demo path
ckpt: "/path/to/Emotion-LLaMA/checkpoints/save_checkpoint/Emotion_LLaMA.pth"
```

### Step 2: Configure Fine-Grained Dataset

Configure the Stage 2 dataset in
`train_configs/minigptv2_tuning_stage_2.yaml`:

```yaml
datasets:
  feature_face_caption:
    task_pool: [reason_v2]
    annotation_format: auto
    build_info:
      image_path: /path/to/MER2023/video
      ann_path: /path/to/MER2023/MERR_fine_grained.txt
      transcription_path: transcription_en_all.csv
      fine_grained_json_path: MERR_fine_grained.json
      face_feature_path: mae_340_UTT
      video_feature_path: maeV_399_UTT
      audio_feature_path: HL-UTT
```

This provides 4,487 samples with detailed multimodal reasoning. Relative
resource paths are resolved from the directory containing `ann_path`. The
transcription CSV is optional but recommended, while all three precomputed
feature trees are required.

### Step 3: Set Advanced Task Instructions

The `task_pool: [reason_v2]` setting in the YAML above selects the advanced
reasoning task. No dataset source edit is needed.

The `reason_v2` task focuses on:
- Detailed multimodal analysis
- Explanation of emotional triggers
- Integration of visual, audio, and textual cues
- Complex emotion reasoning

### Step 4: Retrieve Multimodal Descriptions

`reason_v2` automatically loads the `smp_reason_caption` field from the JSON
file configured by `fine_grained_json_path`.

{: .warning }
> Keep the fine-grained JSON configured during training. Evaluation generation
> uses the instruction input and does not require editing the dataset source.

---

## Training Process

### Run Instruction Tuning

Execute the Stage 2 training script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 train.py --cfg-path train_configs/minigptv2_tuning_stage_2.yaml
```

### Training Configuration

**Key settings in `train_configs/minigptv2_tuning_stage_2.yaml`:**

```yaml
model:
  arch: minigpt_v2
  llama_model: "/path/to/checkpoints/Llama-2-7b-chat-hf"
  ckpt: "/path/to/stage1/checkpoint_29.pth"  # Use Stage 1 checkpoint
  use_grad_checkpoint: True
  chat_template: True
  lora_r: 64
  lora_alpha: 16

datasets:
  feature_face_caption:
    batch_size: 1

run:
  task: image_text_pretrain
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-6      # Lower learning rate for fine-tuning
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

**Stage 2 typically shows:**
- Slower convergence than Stage 1 (more complex reasoning)
- Lower loss values (smaller dataset, more detailed annotations)
- Better qualitative performance on reasoning tasks

**Expected training time**: ~1-2 days on 4x A5000 GPUs

---

## Monitoring Training

### Track Metrics

Monitor these key metrics:

1. **Training Loss**: Should decrease steadily, may plateau around epoch 20-25
2. **Validation Performance**: Evaluate reasoning quality
3. **Generation Quality**: Sample outputs to check reasoning coherence

### Sample Outputs During Training

Periodically check generated outputs:

```python
# Example: Check model output on a validation sample
Input: "What are the facial expressions and vocal tone used in the video?"
Output (Early epochs): "The person shows happiness."
Output (Later epochs): "The person displays a genuine smile with raised cheeks 
(AU06 + AU12), bright upward intonation in voice, and enthusiastic language, 
indicating strong happiness and excitement."
```

---

## Instruction Tuning Best Practices

### 1. Learning Rate

Use a **lower** learning rate than Stage 1:
```yaml
# Stage 1: init_lr: 1e-5
# Stage 2: init_lr: 1e-6  (10x smaller!)
```

This prevents catastrophic forgetting of Stage 1 knowledge.

### 2. Batch Size

Batch size remains 1 per GPU:
```yaml
batch_size: 1  # Same as Stage 1
# With 4 GPUs, effective batch size = 4
```

### 3. Data Augmentation

Enable multimodal augmentation:
```yaml
enable_augmentation: true
augmentation_prob: 0.5
```

### 4. Regularization

Prevent overfitting on small dataset:
```yaml
weight_decay: 0.05
dropout: 0.1
```

---

## Evaluation on EMER Dataset

After training, evaluate emotion reasoning on the EMER benchmark.

### Prepare for Testing

In `eval_configs/eval_emotion_EMER.yaml`, select `task_pool: [reason_v2]` and
configure `fine_grained_json_path` plus the three feature roots. The target
caption is not included in the generation prompt, so no Python edit is needed.

### Configure Evaluation

Specify your trained checkpoint in `eval_configs/eval_emotion_EMER.yaml`:

```yaml
# Set checkpoint path
llama_model: "/path/to/checkpoints/Llama-2-7b-chat-hf"
ckpt: "/path/to/save_checkpoint/2024xxxx-v2/checkpoint_29.pth"
```

### Run Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 eval_emotion_EMER.py --cfg-path eval_configs/eval_emotion_EMER.yaml
```

### EMER Scoring

The EMER dataset evaluates two metrics (0-10 scale):

- **Clue Overlap**: How well the model identifies relevant emotional cues
- **Label Overlap**: Accuracy of emotion label prediction

To score predictions using ChatGPT, refer to:
```
https://github.com/zeroQiaoba/AffectGPT/blob/master/AffectGPT/evaluation.py
```

---

## Expected Results

### EMER Benchmark

| Model | Clue Overlap | Label Overlap |
|:------|:------------:|:-------------:|
| VideoChat-Text | 6.42 | 3.94 |
| Video-LLaMA | 6.64 | 4.89 |
| Valley | 7.24 | 5.77 |
| **Emotion-LLaMA** | **7.83** | **6.25** |

Your trained model should achieve similar or better results.

---

## Advanced Techniques

### Mixed Task Training

Combine multiple reasoning tasks:

```yaml
task_pool:
  - reason
  - reason
  - reason
  - reason_v2
  - reason_v2
  - reason_v2
  - reason_v2
  - reason_v2
  - reason_v2
  - reason_v2
build_info:
  coarse_grained_json_path: MERR_coarse_grained.json
  fine_grained_json_path: MERR_fine_grained.json
```

Repeated task names act as sampling weights; this example gives `reason` and
`reason_v2` a 30/70 split. Mixed reasoning requires both JSON resources.

### Curriculum Learning

Start with simpler samples, gradually increase complexity:

```python
# Sort samples by difficulty (e.g., number of modalities, complexity)
sorted_samples = sort_by_complexity(samples)
```

### Multi-stage Fine-tuning

Fine-tune in multiple sub-stages with different learning rates:

1. **Sub-stage 2.1**: lr=1e-5, epochs 0-15
2. **Sub-stage 2.2**: lr=5e-6, epochs 15-30

---

## Troubleshooting

### Model Generates Generic Responses

**Symptoms**: Model outputs simple labels without reasoning

**Solutions**:
- Decrease learning rate
- Increase training epochs
- Check that fine-grained captions are loaded correctly
- Verify `reason_v2` task is enabled

### Overfitting on Small Dataset

**Symptoms**: Training loss very low, but validation performance poor

**Solutions**:
- Increase weight decay
- Add dropout
- Use data augmentation
- Early stopping based on validation performance

### Inconsistent Reasoning Quality

**Symptoms**: Some outputs are detailed, others are generic

**Solutions**:
- Increase training epochs for better convergence
- Use temperature sampling during inference
- Implement beam search for more stable generation

---

## Checkpoint Management

### Save Best Checkpoints

Configure automatic best-model saving:

```yaml
run:
  save_best: true
  metric: "clue_overlap"  # Save based on EMER metric
```

### Checkpoint Ensemble

Combine multiple checkpoints for better performance:

```python
# Average predictions from checkpoints at epochs 25, 27, 29
checkpoints = [
    "checkpoint_25.pth",
    "checkpoint_27.pth",
    "checkpoint_29.pth"
]
```

---

## Next Steps

After completing instruction tuning:

1. [Evaluate on benchmarks](../evaluation/) - Test on MER2023, EMER, DFEW
2. [Deploy the model](../demo/) - Create demos and applications
3. [Use the API](../api/) - Integrate into your projects

---

## Questions?

For instruction tuning questions:
- Review the [Training Overview](index.md)
- Check [Evaluation Guide](../evaluation/)
- Open an [issue on GitHub](https://github.com/ZebangCheng/Emotion-LLaMA/issues)

