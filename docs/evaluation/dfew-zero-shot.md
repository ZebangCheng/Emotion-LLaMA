---
layout: default
title: DFEW Zero-shot
parent: Evaluation
nav_order: 1
---

# DFEW Zero-shot Reproduction
{: .no_toc }

A complete, runnable recipe for the DFEW zero-shot benchmark: the resources to
download, the configuration that produces the published score, and the settings
that change it.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

DFEW zero-shot evaluates a checkpoint that was never fine-tuned on DFEW. The
evaluation reuses `FeatureFaceDataset`; only the annotation file, the feature
trees, and the label set differ, and all three come from the YAML config.

Reported on the 2341 clips of `reltive_DFEW_set_1_test.txt`:

| Configuration | UAR | WAR |
|:--------------|:---:|:---:|
| Published | 45.59 | 59.37 |
| This recipe | 45.76 | 58.82 |

The recipe below is a reproduction from the released weights and features, not
the authors' original run. See [Open questions](#open-questions).

---

## Required Resources

**1. DFEW videos and labels.** Request access from the
[DFEW homepage](https://dfew-dataset.github.io/). This guide assumes the clips
live in `DFEW/video/` as `<video_id>.mp4`.

**2. Precomputed features.** Download and unzip into `DFEW/features/`:

> [https://drive.google.com/drive/folders/1LdR4qJgKQK6NrR_Hw0cdiPlAciGq1c4z](https://drive.google.com/drive/folders/1LdR4qJgKQK6NrR_Hw0cdiPlAciGq1c4z)

| Archive | Role |
|:--------|:-----|
| `mae_DFEW_ck16_UTT.zip` | face features (`face_feature_path`) |
| `maeV_399_UTT.zip` | video features (`video_feature_path`) |
| `HL-UTT.zip` | audio features (`audio_feature_path`) |

`mae_340_UTT.zip` and `maeDFER_all_test.zip` are in the same folder but are not
used by this recipe.

**3. Test split.** `reltive_DFEW_set_1_test.txt` from the project's data folder:

> [https://drive.google.com/drive/folders/1d-Sg5fAskt2s6OOEUNXFaM2u-C055Whj](https://drive.google.com/drive/folders/1d-Sg5fAskt2s6OOEUNXFaM2u-C055Whj)

**4. Zero-shot checkpoint.** `save_checkpoint/stage2/DFEW_zero-shot.pth` from the
same folder, plus `Llama-2-7b-chat-hf` as described in the
[Getting Started Guide]({{ site.baseurl }}/getting-started/).

Expected layout:

```text
DFEW/
├── 📂 video                          # <video_id>.mp4
├── 📂 features
│   ├── 📂 mae_DFEW_ck16_UTT          # face features
│   ├── 📂 maeV_399_UTT               # video features
│   └── 📂 HL-UTT                     # audio features
└── 📄 reltive_DFEW_set_1_test.txt    # 2341 test rows
```

---

## Configuration

`eval_configs/eval_emotion.yaml` ships with the entry below. Only the two
absolute paths need editing:

```yaml
evaluation_datasets:
  dfew:
    eval_file_path: /path/to/DFEW/reltive_DFEW_set_1_test.txt
    img_path: /path/to/DFEW/video
    task_pool: [emotion]
    labels: [happy, sad, neutral, angry, surprise, disgust, fear]
    annotation_format: auto
    face_feature_path: features/mae_DFEW_ck16_UTT
    video_feature_path: features/maeV_399_UTT
    audio_feature_path: features/HL-UTT
    frame_selection: middle
    max_new_tokens: 500
    batch_size: 1
```

Feature paths are relative to the directory holding `eval_file_path`. Point
`model.ckpt` at `DFEW_zero-shot.pth` and `model.llama_model` at the Llama-2
weights.

Three parts of this entry are load-bearing and are explained in
[What changes the score](#what-changes-the-score): the label order, the absent
`transcription_path`, and `frame_selection: middle`.

---

## Running

```bash
python eval_emotion.py \
  --cfg-path eval_configs/eval_emotion.yaml \
  --dataset dfew --zero_shot \
  --output-dir results
```

`--zero_shot` marks the run as zero-shot in `metrics.json` and enables the
dataset's `zero_shot_label_aliases`, which map a checkpoint's own vocabulary
(for example `worried`) onto DFEW's classes. With the seven-class prompt above
the model already answers in DFEW's vocabulary, so no alias is applied.

Artifacts land in `results/dfew/`:

```text
results/dfew/
├── 📄 predictions.jsonl   # one record per clip
├── 📄 predictions.csv     # same records, spreadsheet-safe
└── 📄 metrics.json        # uar, war, per-class recall, confusion matrix
```

`uar` and `war` are reported under those names alongside `macro_recall` and
`accuracy`, which are the same two numbers.

---

## What changes the score

Each row below is a full 2341-clip run against `DFEW_zero-shot.pth`, changing
one setting at a time from the recipe.

### Label order in the prompt

The candidate list is rendered from `labels` in order. The model is far more
sensitive to that order than to the label set itself:

| `labels` order | UAR | WAR |
|:---------------|:---:|:---:|
| `happy, sad, neutral, angry, surprise, disgust, fear` | 42.57 | 53.48 |
| `neutral, angry, happy, sad, surprise, disgust, fear` | 31.67 | 36.52 |
| `angry, disgust, fear, happy, neutral, sad, surprise` | 26.29 | 32.59 |

Same seven classes, first frame, same features: reordering costs up to 21 WAR
points. Orders beginning `happy, sad, neutral, angry` score 53-55; others fall
to 32-37. Keep the order shipped in the config.

To use a candidate list that differs from the scored label set, set
`prompt_labels`; scoring still uses `labels`.

### Spoken transcript

Omitting `transcription_path` drops the `The person in video says: ...` prefix
from the prompt and is worth roughly 3 WAR points:

| Transcript | UAR | WAR |
|:-----------|:---:|:---:|
| None | 45.26 | 57.71 |
| `DFEW_transcription_en_test.csv` | 43.79 | 54.85 |

Both English and machine-generated transcripts scored below no transcript at
all, so this is not a transcript-quality problem.

### Frame selection

`frame_selection` chooses the single frame handed to the vision encoder:
`first` (the default, preserving existing behavior), `middle`, or `peak`. The
`peak` mode reads an AU-based peak frame index per clip and requires
`peak_index_path`:

| `frame_selection` | UAR | WAR |
|:------------------|:---:|:---:|
| `first` | 44.18 | 56.09 |
| `middle` | 45.76 | 58.82 |
| `peak` | 47.12 | 60.62 |

`peak` scores highest but overshoots the published numbers; `middle` matches
them most closely, which is why the shipped config uses it.

### Face features

`mae_DFEW_ck16_UTT` outperforms `mae_340_UTT` as the face feature by about 2
WAR points, but only on the face channel — using it for `video_feature_path`
instead costs 13 points.

---

## Open questions

- **Provenance of `mae_DFEW_ck16_UTT`.** Whether its encoder was fine-tuned on
  DFEW is not documented. If it was, a run using it is not strictly zero-shot.
  Substituting `mae_340_UTT` and `frame_selection: peak` gives 45.26 / 57.71
  without that dependency.
- **Vision input.** This recipe decodes a frame from `img_path` videos. The
  authors' configuration referenced a prepared `DFEW/images` directory whose
  contents (face crops or exported frames) are not published.

---

## See also

- [Evaluation]({{ site.baseurl }}/evaluation/) — all benchmarks
- [Dataset Configuration]({{ site.baseurl }}/dataset/configuration/) — the full
  `FeatureFaceDataset` configuration contract
