---
layout: default
title: Dataset Configuration
parent: MERR Dataset
nav_order: 2
---

# FeatureFace Dataset Configuration

This page defines the configuration contract for using `FeatureFaceDataset`
without editing `minigpt4/datasets/datasets/first_face.py` for each training or
evaluation run.

## Goals

- Select `emotion`, `reason`, and `reason_v2` tasks from YAML.
- Configure every FeatureFace metadata and precomputed-feature path.
- Load coarse- and fine-grained JSON files only when their tasks need them.
- Support both compact `N E` annotations and legacy `N C E [V]` annotations.
- Keep existing four-argument Python construction and emotion-only behavior.
- Give missing or inconsistent configuration a clear error before training.

## Non-goals

- Converting MER-Factory exports into Emotion-LLaMA annotations.
- Extracting MAE, VideoMAE, or HuBERT features online.
- Refactoring `MER2024Dataset`, which has a separate data layout.
- Changing the nine-label emotion vocabulary or model checkpoints.
- Fixing the legacy bitsandbytes Windows 8-bit loading path.

## Stage 1 training

Filesystem values live under `build_info`. Dataset behavior lives beside the
processor and batching options.

```yaml
datasets:
  feature_face_caption:
    task_pool:
      - emotion
      - reason
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

Absolute paths are used unchanged. Relative metadata and feature paths are
resolved from the directory containing `ann_path`. This allows a dataset folder
to be moved by changing only `ann_path` and `image_path`.

The builder forwards these values explicitly to `FeatureFaceDataset`; it does
not pass the complete OmegaConf object into the dataset.

Stage 1 uses `task_pool: [emotion, reason]`. The `reason` task reads the
`caption` field from `coarse_grained_json_path`; the fine-grained JSON is not
needed for this task pool.

For backward compatibility, the shipped default YAML keeps its existing
`image_path` and `ann_path` values. It adds the behavioral keys and relative
resource names shown above, so users can override every dataset location from a
training config without changing Python source.

## Stage 2 reasoning

Stage 2 switches both the annotation text and reasoning JSON in YAML:

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

`reason_v2` reads `smp_reason_caption` from `fine_grained_json_path`; it does
not load the coarse-grained JSON unless `reason` is also in the task pool.

## Emotion-only custom data

Prepared custom data can run emotion recognition without either reasoning
JSON:

```yaml
datasets:
  feature_face_caption:
    task_pool: [emotion]
    annotation_format: auto
    build_info:
      image_path: /path/to/custom/videos
      ann_path: /path/to/custom/annotations.txt
      transcription_path: transcripts.csv  # optional, but recommended
      face_feature_path: face_features
      video_feature_path: video_features
      audio_feature_path: audio_features
```

The FaceMAE, VideoMAE, and HuBERT feature directories must each contain a
`<video_name>.npy` file for every annotation row. These features must be
prepared before training or evaluation; this configuration change does not add
online feature extraction or MER-Factory conversion.

## Evaluation configuration

The existing evaluation keys remain supported. The same optional resource and
task keys are accepted alongside them:

```yaml
evaluation_datasets:
  feature_face_caption:
    eval_file_path: /path/to/MER2023/relative_test3_NCEV.txt
    img_path: /path/to/MER2023/video
    task_pool:
      - emotion
    annotation_format: auto
    transcription_path: transcription_en_all.csv
    face_feature_path: mae_340_UTT
    video_feature_path: maeV_399_UTT
    audio_feature_path: HL-UTT
```

`eval_emotion.py` and `eval_emotion_EMER.py` forward the optional values while
preserving their current `eval_file_path` and `img_path` interface.

## Python interface

The first four positional arguments remain unchanged. New behavior is exposed
through keyword-only options:

```python
FeatureFaceDataset(
    vis_processor,
    text_processor,
    vis_root,
    ann_path,
    *,
    task_pool=None,
    annotation_format="auto",
    transcription_path=None,
    coarse_grained_json_path=None,
    fine_grained_json_path=None,
    face_feature_path="mae_340_UTT",
    video_feature_path="maeV_399_UTT",
    audio_feature_path="HL-UTT",
)
```

Defaults preserve direct emotion-only construction:

- `task_pool=None` becomes `["emotion"]`.
- no transcript is loaded when `transcription_path` is `None`;
- feature paths default to the existing `mae_340_UTT`, `maeV_399_UTT`, and
  `HL-UTT` directories beside `ann_path`;
- coarse and fine JSON paths must be provided when their corresponding tasks
  are enabled.

All path parameters accept strings and `os.PathLike` values.

## Task-specific resources

| Task | Answer source | Required auxiliary resource |
| --- | --- | --- |
| `emotion` | Emotion label in the annotation row | None |
| `reason` | `caption` field | `coarse_grained_json_path` |
| `reason_v2` | `smp_reason_caption` field | `fine_grained_json_path` |

Only resources required by at least one configured task are opened. A mixed
pool loads the union of its required resources. Task order and duplicates are
preserved so repeated task names can continue to act as sampling weights.

`task_pool` must be a non-empty list or tuple containing only the three names
above. A bare string, an empty collection, or an unknown name raises
`ValueError` with the invalid value and the supported values.

## Optional transcriptions

When `transcription_path` is configured, its CSV must contain `name` and
`sentence` columns. The spoken-text prefix is added to the instruction exactly
as before. Missing files, columns, or sample names produce focused errors.

When it is omitted, the dataset does not read a CSV and emits the instruction
without the `The person in video says: ...` prefix. Transcriptions remain
recommended because textual information is an important model input, but they
are no longer an unconditional initialization requirement.

## Annotation formats

Annotation files are whitespace-delimited. Empty lines are ignored.

`annotation_format: auto` supports both documented forms:

```text
# Compact N E
sample_00000023 angry

# Legacy N C E V
sample_00000023 35 angry -1.174107
```

In `auto` mode, a two-column row uses columns 0 and 1. A row with three or more
columns uses columns 0 and 2. `annotation_format: ne` and
`annotation_format: ncev` enforce one format and reject incompatible rows.

The field letters mean:

- `N`: video name without the `.mp4` or `.avi` extension;
- `C`: frame count retained for legacy annotations;
- `E`: one of the supported emotion labels;
- `V`: optional trailing valence value.

The parsed emotion must remain one of the existing nine labels:
`neutral`, `angry`, `happy`, `sad`, `worried`, `surprise`, `fear`, `contempt`,
or `doubt`.

## Compatibility guarantees

- Registry name `feature_face_caption` and builder output remain unchanged.
- The old four-positional-argument constructor remains valid.
- The default task remains emotion recognition.
- Output keys and instruction markers remain unchanged.
- Feature concatenation stays FaceMAE, VideoMAE, then audio.
- Existing legacy NCEV annotations continue to parse.
- Existing absolute path overrides continue to work.

## Validation and tests

Unit tests use temporary annotation, JSON, CSV, and feature trees. They cover:

1. Emotion-only initialization without either JSON or a transcript.
2. Conditional coarse, fine, and mixed-task resource loading.
3. Relative, absolute, and `PathLike` path resolution.
4. Configured feature directories and feature concatenation order.
5. Builder and evaluation propagation of every optional key.
6. Compact NE and legacy NCEV parsing, including malformed rows.
7. Empty, string, and unknown task pools.
8. Clear errors for missing task resources and transcript columns or samples.

The repository's legacy Python 2 VQA demo prevents a clean Python 3
`compileall` baseline. Verification therefore targets all changed Python files,
the new unit tests, YAML parsing, and whitespace checks.

## Related issues

- [Issue #107](https://github.com/ZebangCheng/Emotion-LLaMA/issues/107):
  config-driven task selection and support for the documented
  compact `N E` annotation form.
- [Issue #128](https://github.com/ZebangCheng/Emotion-LLaMA/issues/128):
  prepared custom datasets can provide their own videos,
  annotations, transcripts, and precomputed feature roots. Export conversion
  and feature extraction remain outside this change.
- [Issue #138](https://github.com/ZebangCheng/Emotion-LLaMA/issues/138):
  training and evaluation inputs become explicit, but online
  demo-style feature extraction remains outside this change.
- [Issue #101](https://github.com/ZebangCheng/Emotion-LLaMA/issues/101): this
  change accepts `PathLike` dataset paths, but it does not fix
  the separately reported legacy bitsandbytes Windows startup failure.

These issues should be referenced rather than automatically closed because each
contains requests beyond this configuration change.
