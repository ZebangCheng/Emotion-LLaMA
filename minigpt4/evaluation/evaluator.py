"""Deterministic, model-independent evaluation and artifact writing.

The functions in this module deliberately avoid importing torch or model code so
metric behavior can be tested without a GPU or checkpoint. Classification
predictions that cannot be mapped to exactly one configured label remain
invalid; they are never rewritten to a valid class.
"""

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata


DEFAULT_EMOTION_LABELS = (
    "neutral",
    "angry",
    "happy",
    "sad",
    "worried",
    "surprise",
    "fear",
    "contempt",
    "doubt",
)
INVALID_LABEL = "__invalid__"
SUPPORTED_EVALUATION_TASKS = ("auto", "classification", "reasoning")


def _normalized_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _validate_labels(labels):
    if labels is None:
        labels = DEFAULT_EMOTION_LABELS
    if isinstance(labels, str):
        raise ValueError("labels must be a non-empty sequence, not a string")

    canonical_labels = []
    normalized = {}
    for raw_label in labels:
        label = str(raw_label).strip()
        normalized_label = _normalized_text(label)
        if not normalized_label:
            raise ValueError("labels cannot contain an empty value")
        if normalized_label in normalized:
            raise ValueError(
                "labels must be unique after normalization: {!r} and {!r}".format(
                    normalized[normalized_label], label
                )
            )
        normalized[normalized_label] = label
        canonical_labels.append(label)

    if not canonical_labels:
        raise ValueError("labels must contain at least one value")
    if INVALID_LABEL in canonical_labels:
        raise ValueError("{} is reserved for invalid predictions".format(INVALID_LABEL))
    return canonical_labels, normalized


class LabelNormalizer:
    """Map generated text to exactly one configured canonical label."""

    def __init__(self, labels=None, aliases=None):
        self.labels, canonical_lookup = _validate_labels(labels)
        self.canonical_lookup = dict(canonical_lookup)
        self.lookup = dict(canonical_lookup)

        aliases = {} if aliases is None else aliases
        if not hasattr(aliases, "items"):
            raise ValueError("label aliases must be a mapping of alias to label")
        canonical_values = set(self.labels)
        for raw_alias, raw_target in aliases.items():
            alias = _normalized_text(raw_alias)
            target_key = _normalized_text(raw_target)
            target = canonical_lookup.get(target_key)
            if target is None or target not in canonical_values:
                raise ValueError(
                    "label alias {!r} points to unknown label {!r}".format(
                        raw_alias, raw_target
                    )
                )
            if not alias:
                raise ValueError("label aliases cannot contain an empty value")
            previous = self.lookup.get(alias)
            if previous is not None and previous != target:
                raise ValueError(
                    "label alias {!r} conflicts with label {!r}".format(
                        raw_alias, previous
                    )
                )
            self.lookup[alias] = target

        self._phrases = sorted(
            self.lookup.items(), key=lambda item: (-len(item[0].split()), -len(item[0]))
        )

    def normalize(self, value):
        """Return ``(canonical_label, invalid_reason)`` for generated text."""
        normalized = _normalized_text(value)
        if not normalized:
            return None, "empty_prediction"

        exact = self.lookup.get(normalized)
        if exact is not None:
            return exact, None

        padded = " {} ".format(normalized)
        matches = {
            target
            for phrase, target in self._phrases
            if " {} ".format(phrase) in padded
        }
        if len(matches) == 1:
            return next(iter(matches)), None
        if len(matches) > 1:
            return None, "ambiguous_prediction"
        return None, "unknown_prediction"

    def normalize_target(self, value):
        """Normalize a ground-truth label, requiring an exact configured value."""
        normalized = _normalized_text(value)
        target = self.canonical_lookup.get(normalized)
        if target is None:
            raise ValueError(
                "target {!r} is not one of the configured labels: {}".format(
                    value, ", ".join(self.labels)
                )
            )
        return target


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def evaluate_classification(records, labels=None, aliases=None):
    """Evaluate classification records without converting invalid predictions."""
    normalizer = LabelNormalizer(labels=labels, aliases=aliases)
    canonical_labels = normalizer.labels
    enriched_records = []

    support = Counter()
    predicted = Counter()
    true_positive = Counter()
    column_labels = canonical_labels + [INVALID_LABEL]
    confusion = {
        target: {prediction: 0 for prediction in column_labels}
        for target in canonical_labels
    }

    invalid_count = 0
    correct_count = 0
    for source_record in records:
        record = dict(source_record)
        if "target" not in record or "prediction" not in record:
            raise ValueError("classification records require target and prediction")

        target = normalizer.normalize_target(record["target"])
        prediction, invalid_reason = normalizer.normalize(record["prediction"])
        prediction_bucket = prediction if prediction is not None else INVALID_LABEL
        is_valid = prediction is not None
        is_correct = is_valid and prediction == target

        support[target] += 1
        confusion[target][prediction_bucket] += 1
        if is_valid:
            predicted[prediction] += 1
        else:
            invalid_count += 1
        if is_correct:
            true_positive[target] += 1
            correct_count += 1

        record.update(
            {
                "schema_version": 1,
                "normalized_target": target,
                "normalized_prediction": prediction,
                "is_valid": is_valid,
                "invalid_reason": invalid_reason,
                "correct": bool(is_correct),
            }
        )
        enriched_records.append(record)

    per_class = {}
    for label in canonical_labels:
        precision = _safe_divide(true_positive[label], predicted[label])
        recall = _safe_divide(true_positive[label], support[label])
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[label]),
            "predicted": int(predicted[label]),
            "true_positive": int(true_positive[label]),
        }

    sample_count = len(enriched_records)
    macro_precision = _mean(
        [per_class[label]["precision"] for label in canonical_labels]
    )
    macro_recall = _mean([per_class[label]["recall"] for label in canonical_labels])
    macro_f1 = _mean([per_class[label]["f1"] for label in canonical_labels])
    weighted_precision = _safe_divide(
        sum(
            per_class[label]["precision"] * support[label]
            for label in canonical_labels
        ),
        sample_count,
    )
    weighted_recall = _safe_divide(
        sum(
            per_class[label]["recall"] * support[label]
            for label in canonical_labels
        ),
        sample_count,
    )
    weighted_f1 = _safe_divide(
        sum(
            per_class[label]["f1"] * support[label]
            for label in canonical_labels
        ),
        sample_count,
    )

    metrics = {
        "schema_version": 1,
        "task": "classification",
        "sample_count": sample_count,
        "labels": list(canonical_labels),
        "accuracy": _safe_divide(correct_count, sample_count),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "invalid_count": invalid_count,
        "invalid_rate": _safe_divide(invalid_count, sample_count),
        "per_class": per_class,
        "confusion_matrix": {
            "row_labels": list(canonical_labels),
            "column_labels": list(column_labels),
            "matrix": [
                [confusion[target][prediction] for prediction in column_labels]
                for target in canonical_labels
            ],
        },
        "primary_metric": "macro_f1",
        "agg_metrics": macro_f1,
    }
    return {"records": enriched_records, "metrics": metrics}


def _reasoning_tokens(value):
    normalized = _normalized_text(value)
    return normalized.split() if normalized else []


def _token_scores(prediction, target):
    prediction_tokens = _reasoning_tokens(prediction)
    target_tokens = _reasoning_tokens(target)
    if not prediction_tokens and not target_tokens:
        return 1.0, 1.0, 1.0
    overlap = sum((Counter(prediction_tokens) & Counter(target_tokens)).values())
    precision = _safe_divide(overlap, len(prediction_tokens))
    recall = _safe_divide(overlap, len(target_tokens))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def evaluate_reasoning(records):
    """Compute transparent lexical metrics while preserving full reasoning text."""
    enriched_records = []
    exact_matches = []
    precisions = []
    recalls = []
    token_f1s = []
    prediction_lengths = []

    for source_record in records:
        record = dict(source_record)
        if "target" not in record or "prediction" not in record:
            raise ValueError("reasoning records require target and prediction")
        normalized_prediction = _normalized_text(record["prediction"])
        normalized_target = _normalized_text(record["target"])
        exact_match = normalized_prediction == normalized_target
        precision, recall, token_f1 = _token_scores(
            record["prediction"], record["target"]
        )
        prediction_length = len(_reasoning_tokens(record["prediction"]))
        exact_matches.append(float(exact_match))
        precisions.append(precision)
        recalls.append(recall)
        token_f1s.append(token_f1)
        prediction_lengths.append(prediction_length)
        record.update(
            {
                "schema_version": 1,
                "normalized_prediction": normalized_prediction,
                "normalized_target": normalized_target,
                "exact_match": bool(exact_match),
                "token_precision": precision,
                "token_recall": recall,
                "token_f1": token_f1,
            }
        )
        enriched_records.append(record)

    metrics = {
        "schema_version": 1,
        "task": "reasoning",
        "sample_count": len(enriched_records),
        "exact_match": _mean(exact_matches),
        "token_precision": _mean(precisions),
        "token_recall": _mean(recalls),
        "token_f1": _mean(token_f1s),
        "average_prediction_tokens": _mean(prediction_lengths),
        "primary_metric": "token_f1",
        "agg_metrics": _mean(token_f1s),
        "note": (
            "Reasoning metrics are lexical diagnostics; use the benchmark's "
            "official semantic evaluator for published EMER scores."
        ),
    }
    return {"records": enriched_records, "metrics": metrics}


def _infer_evaluation_task(records):
    task_names = {str(record.get("task", "")).strip() for record in records}
    task_names.discard("")
    if not task_names or task_names == {"emotion"}:
        return "classification"
    if task_names.issubset({"reason", "reason_v2"}):
        return "reasoning"
    raise ValueError(
        "automatic evaluation requires one task family, got: {}".format(
            ", ".join(sorted(task_names))
        )
    )


def evaluate_records(records, task="auto", labels=None, aliases=None):
    records = list(records)
    if task not in SUPPORTED_EVALUATION_TASKS:
        raise ValueError(
            "task must be one of {}".format(", ".join(SUPPORTED_EVALUATION_TASKS))
        )
    selected_task = _infer_evaluation_task(records) if task == "auto" else task
    if selected_task == "classification":
        return evaluate_classification(records, labels=labels, aliases=aliases)
    return evaluate_reasoning(records)


def _record_identity(record):
    identity = record.get("instance_id")
    if identity is None or str(identity).strip() == "":
        raise ValueError("evaluation records require a non-empty instance_id")
    return str(identity)


def _record_sort_key(record):
    sample_index = record.get("sample_index")
    try:
        sample_index = int(sample_index)
    except (TypeError, ValueError):
        sample_index = math.inf
    return (
        str(record.get("dataset", "")),
        str(record.get("split", "")),
        sample_index,
        _record_identity(record),
    )


def merge_rank_records(rank_records):
    """Merge rank-local records and remove DistributedSampler padding copies.

    Identical records with the same ``instance_id`` are considered sampler
    padding. A conflicting duplicate fails closed rather than hiding a data or
    nondeterminism problem.
    """
    merged = {}
    conflict_fields = (
        "dataset",
        "split",
        "target",
        "prediction",
        "task",
        "sample_id",
        "sample_index",
    )
    for records in rank_records:
        for source_record in records:
            record = dict(source_record)
            identity = _record_identity(record)
            previous = merged.get(identity)
            if previous is None:
                merged[identity] = record
                continue
            conflicts = [
                field
                for field in conflict_fields
                if previous.get(field) != record.get(field)
            ]
            if conflicts:
                raise ValueError(
                    "conflicting duplicate evaluation record {!r}: {}".format(
                        identity, ", ".join(conflicts)
                    )
                )
    return sorted(merged.values(), key=_record_sort_key)


def _atomic_text_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path.parent),
        prefix=".{}-".format(path.name),
        suffix=".tmp",
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _json_dump(value, handle, **kwargs):
    json.dump(value, handle, ensure_ascii=False, allow_nan=False, **kwargs)


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_evaluation_report(report, output_dir):
    """Atomically write JSONL, CSV, and metrics JSON artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(report.get("records", []))
    metrics = dict(report.get("metrics", {}))

    jsonl_path = output_dir / "predictions.jsonl"
    csv_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"

    def write_jsonl(handle):
        for record in records:
            _json_dump(record, handle, sort_keys=True)
            handle.write("\n")

    preferred_fields = [
        "schema_version",
        "instance_id",
        "sample_id",
        "dataset",
        "split",
        "sample_index",
        "task",
        "prediction",
        "normalized_prediction",
        "target",
        "normalized_target",
        "is_valid",
        "invalid_reason",
        "correct",
        "exact_match",
        "token_precision",
        "token_recall",
        "token_f1",
    ]
    present_fields = {key for record in records for key in record}
    fieldnames = [field for field in preferred_fields if field in present_fields]
    fieldnames.extend(sorted(present_fields - set(fieldnames)))

    def write_csv(handle):
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(value) for key, value in record.items()})

    _atomic_text_write(jsonl_path, write_jsonl)
    _atomic_text_write(csv_path, write_csv)
    _atomic_text_write(
        metrics_path,
        lambda handle: _json_dump(metrics, handle, indent=2, sort_keys=True),
    )
    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "metrics": str(metrics_path),
    }
