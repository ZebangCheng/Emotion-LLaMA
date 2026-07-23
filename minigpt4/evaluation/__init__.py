"""Shared evaluation utilities for emotion classification and reasoning."""

from .evaluator import (
    DEFAULT_EMOTION_LABELS,
    INVALID_LABEL,
    evaluate_classification,
    evaluate_reasoning,
    evaluate_records,
    merge_rank_records,
    write_evaluation_report,
)
from .tracker import (
    ValidationTracker,
    collapse_single_eval_datasets,
    resolve_best_checkpoint_path,
    validate_split_configuration,
)

__all__ = [
    "DEFAULT_EMOTION_LABELS",
    "INVALID_LABEL",
    "ValidationTracker",
    "collapse_single_eval_datasets",
    "evaluate_classification",
    "evaluate_reasoning",
    "evaluate_records",
    "merge_rank_records",
    "resolve_best_checkpoint_path",
    "write_evaluation_report",
    "validate_split_configuration",
]
