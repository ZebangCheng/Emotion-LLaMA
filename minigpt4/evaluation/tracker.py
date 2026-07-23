"""Validation metric tracking independent of the training runtime."""

import math
import numbers
from pathlib import Path


SPLIT_GROUP_NAMES = ("train", "validation", "test")


def resolve_best_checkpoint_path(
    resume_checkpoint_path=None,
    stored_best_checkpoint_path=None,
    output_dir=None,
):
    """Find an existing best checkpoint when a run resumes in a new job dir."""
    candidates = []
    if resume_checkpoint_path:
        resume_path = Path(resume_checkpoint_path)
        candidates.append(resume_path.parent / "checkpoint_best.pth")
    if stored_best_checkpoint_path:
        candidates.append(Path(stored_best_checkpoint_path))
    if output_dir:
        candidates.append(Path(output_dir) / "checkpoint_best.pth")

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        candidate_key = str(candidate.resolve())
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def validate_split_configuration(
    available_splits,
    train_splits,
    valid_splits,
    test_splits,
    evaluate_only=False,
    best_model_split=None,
):
    """Validate explicit split roles without inferring validation from test data."""
    available = set(available_splits)
    groups = {
        "train": list(train_splits),
        "validation": list(valid_splits),
        "test": list(test_splits),
    }
    for group_name, split_names in groups.items():
        if len(set(split_names)) != len(split_names):
            raise ValueError("{} splits contain duplicates".format(group_name))
        missing = [split for split in split_names if split not in available]
        if missing:
            raise ValueError(
                "configured {} split(s) are missing from the datasets: {}".format(
                    group_name, ", ".join(missing)
                )
            )

    for left_index, left_name in enumerate(SPLIT_GROUP_NAMES):
        for right_name in SPLIT_GROUP_NAMES[left_index + 1 :]:
            overlap = set(groups[left_name]) & set(groups[right_name])
            if overlap:
                raise ValueError(
                    "dataset split(s) cannot be both {} and {}: {}".format(
                        left_name, right_name, ", ".join(sorted(overlap))
                    )
                )

    if not evaluate_only and len(groups["train"]) != 1:
        raise ValueError("training requires exactly one configured train split")
    if evaluate_only and not groups["test"]:
        raise ValueError(
            "evaluate-only mode requires at least one explicitly configured test split"
        )

    if groups["validation"]:
        primary_split = best_model_split or groups["validation"][0]
        if primary_split not in groups["validation"]:
            raise ValueError(
                "best_model_split {!r} is not in valid_splits".format(primary_split)
            )
    else:
        if best_model_split is not None:
            raise ValueError("best_model_split requires at least one valid split")
        primary_split = None
    return primary_split


def collapse_single_eval_datasets(datasets, batch_sizes, train_splits):
    """Collapse one-element val/test lists and reject ambiguous multi-dataset metrics."""
    datasets = dict(datasets)
    batch_sizes = dict(batch_sizes)
    train_splits = set(train_splits)
    for split_name in list(datasets):
        if split_name in train_splits:
            continue
        split_datasets = datasets[split_name]
        split_batch_sizes = batch_sizes[split_name]
        if not isinstance(split_datasets, (list, tuple)):
            continue
        if len(split_datasets) != 1:
            raise ValueError(
                "validation/test split {!r} requires exactly one dataset, got {}".format(
                    split_name, len(split_datasets)
                )
            )
        datasets[split_name] = split_datasets[0]
        if isinstance(split_batch_sizes, (list, tuple)):
            batch_sizes[split_name] = split_batch_sizes[0]
    return datasets, batch_sizes


class ValidationTracker:
    """Track the best validation metric and an optional early-stop patience."""

    def __init__(
        self,
        metric_name="agg_metrics",
        greater_is_better=True,
        patience=None,
        min_delta=0.0,
    ):
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("metric_name must be a non-empty string")
        if patience is not None:
            if isinstance(patience, bool) or int(patience) != patience or patience < 1:
                raise ValueError("early_stopping_patience must be null or at least 1")
            patience = int(patience)
        min_delta = float(min_delta)
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("early_stopping_min_delta must be finite and non-negative")

        self.metric_name = metric_name
        self.greater_is_better = bool(greater_is_better)
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.best_epoch = None
        self.bad_epochs = 0

    @property
    def has_best(self):
        return self.best_metric is not None

    def _coerce_metric(self, value):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                "validation metric {!r} must be a finite number".format(
                    self.metric_name
                )
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                "validation metric {!r} must be a finite number".format(
                    self.metric_name
                )
            )
        return value

    def update(self, value, epoch):
        value = self._coerce_metric(value)
        if self.best_metric is None:
            improved = True
        elif self.greater_is_better:
            improved = value > self.best_metric + self.min_delta
        else:
            improved = value < self.best_metric - self.min_delta

        if improved:
            self.best_metric = value
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = self.patience is not None and self.bad_epochs >= self.patience
        return {"improved": improved, "should_stop": should_stop}

    def state_dict(self):
        return {
            "metric_name": self.metric_name,
            "greater_is_better": self.greater_is_better,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state):
        if not state:
            return
        if state.get("metric_name", self.metric_name) != self.metric_name:
            raise ValueError(
                "checkpoint tracks metric {!r}, current run tracks {!r}".format(
                    state.get("metric_name"), self.metric_name
                )
            )
        checkpoint_direction = bool(
            state.get("greater_is_better", self.greater_is_better)
        )
        if checkpoint_direction != self.greater_is_better:
            raise ValueError("checkpoint greater_is_better does not match current run")

        best_metric = state.get("best_metric")
        self.best_metric = (
            None if best_metric is None else self._coerce_metric(best_metric)
        )
        best_epoch = state.get("best_epoch")
        self.best_epoch = None if best_epoch is None else int(best_epoch)
        self.bad_epochs = int(state.get("bad_epochs", 0))
        if self.bad_epochs < 0:
            raise ValueError("checkpoint bad_epochs cannot be negative")
