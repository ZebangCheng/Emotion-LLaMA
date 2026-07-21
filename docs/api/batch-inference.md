---
layout: default
title: Batch Inference
parent: API Documentation
nav_order: 3
---

# Batch Inference
{: .no_toc }

Run Emotion-LLaMA from a terminal with one reusable model runtime, durable
JSONL results, and safe resume behavior.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Before You Start

Complete the repository's normal environment and checkpoint setup first. The
default configuration expects these local assets:

- Llama 2 under `checkpoints/Llama-2-7b-chat-hf`
- Emotion-LLaMA at `checkpoints/save_checkpoint/Emoation_LLaMA.pth`
- HuBERT at `checkpoints/transformer/chinese-hubert-large`

The shipped configuration uses an 8-bit Llama model. Select a logical CUDA
device with `--device cuda` or `--device cuda:N`; the runtime places the
quantized weights on that device during construction and moves only the
remaining model components afterward. `--device cpu` is rejected while
`model.low_resource=true`; set `model.low_resource=false` only when the full
precision model fits the selected device.

The command processes records sequentially. It reuses the Emotion-LLaMA,
visual processor, and HuBERT instances, but it does not combine videos into a
tensor batch.

---

## Quick Start

Create `videos.jsonl`:

```json
{"id":"sample-001","video":"videos/clip 01.mp4","prompt":"What emotion is expressed, and why?"}
{"id":"sample-002","video":"videos/clip-02.mp4","prompt":"Describe the speaker's emotion."}
```

Run the manifest and resume safely after an interruption:

```bash
python batch_infer.py \
  --input videos.jsonl \
  --output predictions.jsonl \
  --resume
```

The model stack is initialized once, and one JSON object is persisted after
each record.

---

## Manifest Formats

Both CSV and JSONL are supported. Files are decoded as UTF-8, including UTF-8
BOM files commonly produced on Windows.

| Field | Required | Description |
|:------|:---------|:------------|
| `video` | Yes | Video path. Relative paths use the manifest directory by default. |
| `prompt` | Per row or globally | Prompt for this video. Use `--prompt` or `--prompt-file` as a fallback. |
| `id` | No | Stable unique ID. When omitted, the normalized `video` value is used. |

IDs must be non-empty and unique. Malformed CSV, overflow columns, invalid
JSON, and duplicate JSON object keys are rejected before the model is loaded.

### CSV

```csv
id,video,prompt
sample-001,videos/clip-01.mp4,"What emotion is expressed, and why?"
sample-002,videos/clip-02.mp4,Describe the speaker's emotion.
```

```bash
python batch_infer.py --input videos.csv --output predictions.jsonl
```

### Global Prompt and Base Directory

Use one prompt for rows that omit `prompt`:

```bash
python batch_infer.py \
  --input manifests/videos.jsonl \
  --base-dir D:/datasets/MER/videos \
  --prompt "What emotion is expressed?" \
  --output predictions.jsonl
```

`--force-prompt` replaces every row-level prompt with the global prompt.

---

## Single Video and Directory Modes

Analyze one video:

```bash
python batch_infer.py \
  --video "D:/videos/clip 01.mp4" \
  --id sample-001 \
  --prompt "What emotion is expressed?" \
  --output prediction.jsonl
```

Scan a directory in deterministic path order:

```bash
python batch_infer.py \
  --video-dir videos \
  --recursive \
  --extensions .mp4,.avi,.mov,.mkv \
  --prompt "What emotion is expressed?" \
  --output predictions.jsonl
```

Paths with spaces and Unicode names are supported. Quote paths in the shell.
Directory IDs use forward-slash relative paths so that output remains stable
on Windows.

---

## Output Schema

The output is UTF-8 JSONL with one unique record per ID. A success record has
this schema:

```json
{
  "schema_version": 1,
  "run_fingerprint": "sha256:...",
  "request_hash": "sha256:...",
  "video_fingerprint": "sha256:...",
  "id": "sample-001",
  "video": "videos/clip 01.mp4",
  "resolved_video": "D:\\project\\videos\\clip 01.mp4",
  "prompt": "What emotion is expressed, and why?",
  "source": {
    "mode": "jsonl",
    "input": "D:\\project\\videos.jsonl",
    "record": 1
  },
  "status": "success",
  "answer": "The speaker appears happy because ...",
  "error": null,
  "duration_ms": 1250
}
```

For a failed item, `status` is `error`, `answer` is `null`, and `error`
contains stable diagnostic fields:

```json
{
  "stage": "inference",
  "type": "RuntimeError",
  "message": "Failed to decode the video"
}
```

| Field | Meaning |
|:------|:--------|
| `schema_version` | Output contract version. Unsupported versions are rejected on resume. |
| `run_fingerprint` | Hash of the selected/default model config, resolved device, CLI generation settings, seed, local model-asset contents, `batch_infer.py`, and all Python/YAML files under `minigpt4`. |
| `request_hash` | Hash of ID, resolved video path, and prompt. |
| `video_fingerprint` | SHA-256 of the complete video contents. Pending videos are checked again before and after inference; the field may be `null` for a missing-file error. |
| `source` | Input mode, source path, and record number. |
| `duration_ms` | Wall-clock duration for this attempt, rounded to milliseconds. |

---

## Resume and Output Safety

`--resume` applies the following rules:

- Matching `success` records are skipped.
- `error` records are retried and atomically replaced by ID.
- Unattempted error records remain intact after `--fail-fast`, out-of-memory,
  or interruption.
- A prior success conflicts when its prompt, path, video contents, resolved
  device, local checkpoint/Llama/HuBERT contents, runtime settings, or the
  tracked `minigpt4` source/config snapshot changes instead of silently
  reusing an old answer. Prior errors may observe new video contents so that a
  missing file can be added and retried.
- An invalid final JSONL fragment without a line terminator is treated as
  truncated, discarded, and reprocessed. A terminated corrupt line or
  corruption in the middle of the file is rejected.
- Every historical output ID must still exist in the current input manifest.

The command takes a non-blocking sidecar lock named
`.OUTPUT_FILENAME.lock` for the complete run. A second process targeting the
same output exits instead of racing the first process. The sidecar file may
remain after a successful run; the operating-system lock, not file existence,
indicates ownership.

Use `--overwrite` to start the selected input again. Input validation and
runtime initialization happen before an existing output is replaced, so a
model-loading failure preserves the old file.

{: .note }
> Strict resume fingerprints stream the complete contents of every input video
> and local checkpoint, Llama, and HuBERT artifact. Large model directories can
> therefore add noticeable startup I/O. Local model contents are verified again
> after runtime initialization so an in-place replacement cannot mix identities
> within one output. Remote model identifiers cannot expose a stable local
> content snapshot; use pinned local model directories when reproducible resume
> behavior is required.

Configuration values and `--options` used by the batch command must be
concrete. OmegaConf interpolation such as `${oc.env:MODEL_PATH}` is rejected
because its resolved value can change without changing the YAML or command
text; resolve the value before invoking `batch_infer.py`. Declare `model.arch`
and `model.model_type` in YAML rather than overriding them through `--options`,
so runtime loading and asset fingerprinting select the same default config.

---

## Failure Isolation and Exit Codes

An individual missing, unreadable, or invalid video produces an `error`
record and processing continues. Use `--fail-fast` to stop after the first
item error.

| Exit code | Meaning |
|:----------|:--------|
| `0` | Every attempted item succeeded, or all matching successes were skipped. |
| `1` | One or more item-level errors were written. |
| `2` | Invalid arguments, manifest, prompt, ID, or resume conflict. |
| `3` | Shared runtime initialization failed, or inference stopped after an out-of-memory error. |
| `4` | Output is corrupt, locked, unreadable, or cannot be updated safely. |
| `130` | Interrupted by the user. Completed records remain durable. |

Run `python batch_infer.py --help` for generation controls such as
`--temperature`, `--max-new-tokens`, `--num-beams`, `--top-p`, and `--seed`.
The per-item seed is derived from the global seed and request hash, so resume
order does not change sampling for a record.

---

## Reuse the Runtime from Python

The same lazy runtime is importable by another frontend:

```python
from minigpt4.inference import EmotionLLaMARuntime

runtime = EmotionLLaMARuntime("eval_configs/demo.yaml", device="cuda:0")
runtime.load_audio_encoder()

answer = runtime.analyze(
    "videos/sample.mp4",
    "What emotion is expressed?",
    seed=42,
)
print(answer)
```

Construction is lazy. `load()`, `load_audio_encoder()`, and repeated
`analyze()` calls reuse the same components, while every analysis receives a
fresh conversation state. Calls on one runtime are serialized because the
shared model and generation state are not thread-safe.
