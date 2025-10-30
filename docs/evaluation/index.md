---
layout: default
title: Evaluation
nav_order: 5
permalink: /evaluation/
---

# Evaluation
{: .no_toc }

Comprehensive guide to evaluating Emotion-LLaMA on benchmark datasets.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Emotion-LLaMA has been evaluated on multiple benchmark datasets:

- **MER2023 Challenge** - Multimodal emotion recognition
- **EMER Dataset** - Emotion reasoning evaluation
- **MER2024 Challenge** - Noise-robust recognition
- **DFEW** - Zero-shot evaluation

---

## MER2023 Challenge

### Performance Results

Emotion-LLaMA achieves state-of-the-art performance on the MER2023 challenge:

| Method | Modality | F1 Score |
|:-------|:--------:|:--------:|
| wav2vec 2.0 | A | 0.4028 |
| VGGish | A | 0.5481 |
| HuBERT | A | 0.8511 |
| ResNet | V | 0.4132 |
| MAE | V | 0.5547 |
| VideoMAE | V | 0.6068 |
| RoBERTa | T | 0.4061 |
| BERT | T | 0.4360 |
| MacBERT | T | 0.4632 |
| MER2023-Baseline | A, V | 0.8675 |
| MER2023-Baseline | A, V, T | 0.8640 |
| Transformer | A, V, T | 0.8853 |
| FBP | A, V, T | 0.8855 |
| VAT | A, V | 0.8911 |
| Emotion-LLaMA (ours) | A, V | 0.8905 |
| **Emotion-LLaMA (ours)** | **A, V, T** | **0.9036** |

### Evaluation Setup

**Step 1**: Configure the checkpoint path in `eval_configs/eval_emotion.yaml`:

```yaml
# Set pretrained checkpoint path
llama_model: "/path/to/checkpoints/Llama-2-7b-chat-hf"
ckpt: "/path/to/checkpoints/save_checkpoint/stage2/checkpoint_best.pth"
```

**Step 2**: Run the evaluation script:

```bash
torchrun --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset feature_face_caption
```

**Step 3**: View results in `checkpoints/save_checkpoint/stage2/result/MER2023.txt`

### Metrics Explained

**F1 Score**: Harmonic mean of precision and recall

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

For multiclass:
```
F1_macro = average(F1_class1, F1_class2, ..., F1_classN)
```

---

## EMER Dataset

### Performance Results

Emotion-LLaMA excels in emotion reasoning:

| Models | Clue Overlap | Label Overlap |
|:-------|:------------:|:-------------:|
| VideoChat-Text | 6.42 | 3.94 |
| Video-LLaMA | 6.64 | 4.89 |
| Video-ChatGPT | 6.95 | 5.74 |
| PandaGPT | 7.14 | 5.51 |
| VideoChat-Embed | 7.15 | 5.65 |
| Valley | 7.24 | 5.77 |
| **Emotion-LLaMA (ours)** | **7.83** | **6.25** |

### Evaluation Setup

**Step 1**: Set checkpoint path in `eval_configs/eval_emotion_EMER.yaml`:

```yaml
# Set checkpoint path
llama_model: "/path/to/checkpoints/Llama-2-7b-chat-hf"
ckpt: "/path/to/save_checkpoint/stage2/checkpoint_best.pth"
```

**Step 2**: Configure for testing (in `minigpt4/datasets/datasets/first_face.py`):

```python
# Disable caption during testing
# caption = self.fine_grained_dict[video_name]['smp_reason_caption']
caption = ""  # for test reasoning
```

**Step 3**: Run evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 eval_emotion_EMER.py --cfg-path eval_configs/eval_emotion_EMER.yaml
```

### EMER Metrics

**Clue Overlap** (0-10 scale):
- Measures how well the model identifies relevant emotional cues
- Compares generated clues with ground truth
- Higher scores indicate better multimodal understanding

**Label Overlap** (0-10 scale):
- Measures emotion label prediction accuracy
- Accounts for both exact and similar emotion matches
- Higher scores indicate better classification

### Scoring with ChatGPT

Use the AffectGPT evaluation script to score predictions:

```python
# Reference: https://github.com/zeroQiaoba/AffectGPT/blob/master/AffectGPT/evaluation.py

import openai

def score_emer_predictions(predictions, ground_truth):
    """
    Score EMER predictions using ChatGPT
    Returns: clue_overlap, label_overlap
    """
    # Implementation based on AffectGPT
    pass
```

---

## MER2024 Challenge

### Performance Results

Emotion-LLaMA achieved outstanding results in MER2024:

#### MER-NOISE Track (Championship 🏆)

Our team **SZTU-CMU** won with F1 = **0.8530**:

| Teams | Score |
|:------|:-----:|
| **SZTU-CMU** | **0.8530** (1st) |
| BZL arc06 | 0.8383 (2nd) |
| VIRlab | 0.8365 (3rd) |
| T_MERG | 0.8271 (4th) |
| AI4AI | 0.8128 (5th) |

{: .note }
> Emotion-LLaMA achieved **F1 = 0.8452** as a base model, with Conv-Attention enhancement reaching 0.8530.

#### MER-OV Track (3rd Place 🥉)

Emotion-LLaMA scored highest among all individual models:

- **UAR**: 45.59
- **WAR**: 59.37

### Evaluation Setup

**Step 1**: Configure checkpoint for MER2024:

```yaml
# Set pretrained checkpoint path
llama_model: "/path/to/checkpoints/Llama-2-7b-chat-hf"
ckpt: "/path/to/checkpoints/save_checkpoint/stage2/MER2024-best.pth"
```

**Step 2**: Run evaluation on MER2024-NOISE:

```bash
torchrun --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset mer2024_caption
```

---

## DFEW Dataset (Zero-shot)

### Performance Results

Zero-shot evaluation on DFEW (Dynamic Facial Expression in the Wild):

| Method | UAR | WAR |
|:-------|:---:|:---:|
| Baseline | 38.12 | 52.34 |
| Video-LLaMA | 41.23 | 55.67 |
| **Emotion-LLaMA** | **45.59** | **59.37** |

**UAR**: Unweighted Average Recall (balanced accuracy)  
**WAR**: Weighted Average Recall (overall accuracy)

### Zero-shot Evaluation

Emotion-LLaMA demonstrates strong generalization without fine-tuning on DFEW:

```bash
# No fine-tuning needed - direct evaluation
torchrun --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset dfew --zero_shot
```

---

## Custom Evaluation

### Evaluate on Your Own Data

**Step 1**: Prepare your dataset in the same format:

```
video_name, emotion_label, transcription
sample_001.mp4, happiness, "I'm so happy today!"
sample_002.mp4, sadness, "This is really disappointing."
```

**Step 2**: Create a dataset configuration:

```yaml
# custom_dataset.yaml
datasets:
  custom_eval:
    vis_processor:
      train:
        name: "blip2_video_train"
    text_processor:
      train:
        name: "blip_caption"
    
    annotation_file: "/path/to/your/annotations.txt"
    video_path: "/path/to/your/videos/"
```

**Step 3**: Run evaluation:

```bash
torchrun --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/custom_eval.yaml --dataset custom_eval
```

---

## Evaluation Metrics

### Classification Metrics

**Accuracy**:
```
Accuracy = Correct Predictions / Total Predictions
```

**Precision** (per class):
```
Precision = True Positives / (True Positives + False Positives)
```

**Recall** (per class):
```
Recall = True Positives / (True Positives + False Negatives)
```

**F1 Score**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Reasoning Metrics

**Clue Overlap**: Semantic similarity between generated and ground-truth emotional cues

**Label Overlap**: Agreement on emotion labels with partial credit for similar emotions

**BLEU/ROUGE**: Text generation quality metrics

---

## Benchmark Comparison

### Overall Performance

| Benchmark | Metric | Score | Rank |
|:----------|:-------|:-----:|:----:|
| MER2023 | F1 Score | 0.9036 | 1st |
| EMER | Clue Overlap | 7.83 | 1st |
| EMER | Label Overlap | 6.25 | 1st |
| MER2024-NOISE | F1 Score | 0.8530 | 1st |
| MER2024-OV | UAR | 45.59 | 3rd* |
| DFEW (zero-shot) | WAR | 59.37 | - |

\* Highest among individual models (without ensemble)

---

## Interpreting Results

### Good Performance Indicators

✅ **F1 > 0.85** on MER datasets  
✅ **Clue Overlap > 7.0** on EMER  
✅ **Label Overlap > 6.0** on EMER  
✅ **Balanced performance** across emotion categories  

### Common Issues

❌ **Low precision**: Model over-predicts certain emotions
- Solution: Adjust classification threshold or retrain with balanced data

❌ **Low recall**: Model misses certain emotions
- Solution: Augment training data for underrepresented emotions

❌ **Low clue overlap**: Reasoning doesn't match ground truth
- Solution: More instruction tuning on fine-grained data

---

## Reproducing Results

To reproduce our published results:

1. **Use the exact same checkpoints** from our releases
2. **Follow the preprocessing** steps precisely
3. **Use identical evaluation scripts** without modifications
4. **Set random seeds** for deterministic results:
   ```yaml
   seed: 42
   ```

Expected variance: ±0.2% F1 score due to hardware differences

---

## Publication Results

For citing our results in your research, use these exact numbers:

**MER2023 Challenge**:
- F1 Score (A, V, T): **0.9036**

**EMER Dataset**:
- Clue Overlap: **7.83**
- Label Overlap: **6.25**

**MER2024 Challenge**:
- MER-NOISE F1: **0.8530** (with Conv-Attention)
- MER-OV UAR: **45.59**

---

## Next Steps

- [Deploy your model](../demo/) after evaluation
- [Use the API](../api/) for inference
- [Train on custom data](../training/) to improve specific metrics

---

## Questions?

For evaluation-related questions:
- Review the [evaluation scripts](https://github.com/ZebangCheng/Emotion-LLaMA/tree/main/eval_configs)
- Check [training documentation](../training/)
- Open an [issue on GitHub](https://github.com/ZebangCheng/Emotion-LLaMA/issues)

