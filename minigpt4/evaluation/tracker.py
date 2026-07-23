"""Validation metric tracking independent of the training runtime."""

import math
import numbers


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
        if bool(state.get("greater_is_better", self.greater_is_better)) != self.greater_is_better:
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
