import ast
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = (
    REPOSITORY_ROOT / "minigpt4" / "evaluation" / "evaluator.py"
)
TRACKER_PATH = REPOSITORY_ROOT / "minigpt4" / "evaluation" / "tracker.py"
CLI_PATH = REPOSITORY_ROOT / "minigpt4" / "evaluation" / "cli.py"


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_legacy_reasoning_csv_writer(csv_safe_value):
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    writer_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_write_legacy_reasoning_csv"
    )
    namespace = {
        "csv": csv,
        "csv_safe_value": csv_safe_value,
        "os": os,
        "Path": Path,
        "tempfile": tempfile,
    }
    exec(
        compile(
            ast.Module(body=[writer_function], type_ignores=[]),
            str(CLI_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["_write_legacy_reasoning_csv"]


class ClassificationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_source_module("emotion_evaluator_under_test", EVALUATOR_PATH)

    def test_unknown_empty_and_ambiguous_predictions_remain_invalid(self):
        report = self.evaluator.evaluate_classification(
            [
                {"instance_id": "0", "target": "neutral", "prediction": "angry"},
                {"instance_id": "1", "target": "neutral", "prediction": ""},
                {
                    "instance_id": "2",
                    "target": "neutral",
                    "prediction": "happy and sad",
                },
            ],
            labels=["neutral", "happy", "sad"],
        )

        self.assertEqual(report["metrics"]["invalid_count"], 3)
        self.assertEqual(report["metrics"]["accuracy"], 0.0)
        self.assertEqual(
            [record["invalid_reason"] for record in report["records"]],
            ["unknown_prediction", "empty_prediction", "ambiguous_prediction"],
        )
        self.assertTrue(
            all(record["normalized_prediction"] is None for record in report["records"])
        )

    def test_normalization_uses_token_boundaries_and_configured_aliases(self):
        report = self.evaluator.evaluate_classification(
            [
                {
                    "instance_id": "0",
                    "target": "happy",
                    "prediction": "The emotion is JOY!",
                },
                {
                    "instance_id": "1",
                    "target": "happy",
                    "prediction": "unhappy",
                },
            ],
            labels=["happy"],
            aliases={"joy": "happy"},
        )

        self.assertEqual(report["records"][0]["normalized_prediction"], "happy")
        self.assertTrue(report["records"][0]["correct"])
        self.assertEqual(
            report["records"][1]["invalid_reason"], "unknown_prediction"
        )

    def test_metrics_and_invalid_confusion_bucket_match_known_example(self):
        report = self.evaluator.evaluate_classification(
            [
                {"instance_id": "0", "target": "happy", "prediction": "Happy!"},
                {
                    "instance_id": "1",
                    "target": "sad",
                    "prediction": "I think sad.",
                },
                {
                    "instance_id": "2",
                    "target": "neutral",
                    "prediction": "garbled output",
                },
                {"instance_id": "3", "target": "happy", "prediction": "sad"},
            ],
            labels=["happy", "sad", "neutral"],
        )
        metrics = report["metrics"]

        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["war"], metrics["accuracy"])
        self.assertAlmostEqual(metrics["uar"], metrics["macro_recall"])
        self.assertAlmostEqual(metrics["macro_f1"], 4.0 / 9.0)
        self.assertAlmostEqual(metrics["weighted_f1"], 0.5)
        self.assertAlmostEqual(metrics["invalid_rate"], 0.25)
        self.assertEqual(
            metrics["confusion_matrix"]["column_labels"],
            ["happy", "sad", "neutral", "__invalid__"],
        )
        self.assertEqual(
            sum(sum(row) for row in metrics["confusion_matrix"]["matrix"]), 4
        )

    def test_label_contract_rejects_duplicates_and_bad_alias_targets(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.evaluator.evaluate_classification([], labels=["Happy", " happy "])
        with self.assertRaisesRegex(ValueError, "unknown label"):
            self.evaluator.evaluate_classification(
                [], labels=["happy"], aliases={"joy": "neutral"}
            )
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.evaluator.evaluate_classification([], labels=["__INVALID__"])
        with self.assertRaisesRegex(ValueError, "target"):
            self.evaluator.evaluate_classification(
                [{"instance_id": "0", "target": "joy", "prediction": "happy"}],
                labels=["happy"],
                aliases={"joy": "happy"},
            )

    def test_empty_input_has_finite_zero_metrics(self):
        report = self.evaluator.evaluate_classification([], labels=["happy", "sad"])
        metrics = report["metrics"]

        self.assertEqual(metrics["sample_count"], 0)
        for key in ("accuracy", "macro_f1", "weighted_f1", "invalid_rate"):
            self.assertEqual(metrics[key], 0.0)
            self.assertTrue(math.isfinite(metrics[key]))


class ReasoningEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_source_module("reason_evaluator_under_test", EVALUATOR_PATH)

    def test_reasoning_reports_exact_match_and_lexical_token_f1(self):
        report = self.evaluator.evaluate_reasoning(
            [
                {
                    "instance_id": "0",
                    "target": "Raised voice and a frown",
                    "prediction": "raised voice and a frown!",
                },
                {
                    "instance_id": "1",
                    "target": "quiet speech",
                    "prediction": "quiet",
                },
            ]
        )

        self.assertEqual(report["metrics"]["exact_match"], 0.5)
        self.assertAlmostEqual(report["records"][1]["token_precision"], 1.0)
        self.assertAlmostEqual(report["records"][1]["token_recall"], 0.5)
        self.assertAlmostEqual(report["records"][1]["token_f1"], 2.0 / 3.0)
        self.assertIn("lexical", report["metrics"]["note"].lower())

    def test_auto_task_rejects_mixed_classification_and_reasoning(self):
        records = [
            {"instance_id": "0", "task": "emotion", "target": "happy", "prediction": "happy"},
            {"instance_id": "1", "task": "reason", "target": "because", "prediction": "because"},
        ]
        with self.assertRaisesRegex(ValueError, "one task family"):
            self.evaluator.evaluate_records(records, task="auto", labels=["happy"])


class DistributedMergeAndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_source_module("artifact_evaluator_under_test", EVALUATOR_PATH)

    def test_rank_merge_deduplicates_padding_and_restores_sample_order(self):
        sample_zero = {
            "instance_id": "dataset:val:0",
            "dataset": "dataset",
            "split": "val",
            "sample_index": 0,
            "sample_id": "zero",
            "task": "emotion",
            "target": "happy",
            "prediction": "happy",
        }
        sample_one = dict(sample_zero, instance_id="dataset:val:1", sample_index=1, sample_id="one")

        merged = self.evaluator.merge_rank_records(
            [[sample_one, sample_zero], [dict(sample_zero)]]
        )

        self.assertEqual([record["sample_index"] for record in merged], [0, 1])

    def test_rank_merge_fails_closed_on_conflicting_duplicate(self):
        first = {
            "instance_id": "same",
            "sample_id": "sample",
            "sample_index": 0,
            "task": "emotion",
            "target": "happy",
            "prediction": "happy",
        }
        second = dict(first, prediction="sad")
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            self.evaluator.merge_rank_records([[first], [second]])

    def test_report_writes_utf8_jsonl_csv_and_metrics_atomically(self):
        report = self.evaluator.evaluate_reasoning(
            [
                {
                    "instance_id": "0",
                    "sample_id": "样本",
                    "target": "语气 平静",
                    "prediction": "语气 平静",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.evaluator.write_evaluation_report(report, temporary_directory)

            jsonl_record = json.loads(
                Path(paths["jsonl"]).read_text(encoding="utf-8").splitlines()[0]
            )
            metrics = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
            with Path(paths["csv"]).open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(jsonl_record["sample_id"], "样本")
            self.assertEqual(rows[0]["sample_id"], "样本")
            self.assertEqual(metrics["sample_count"], 1)
            self.assertFalse(list(Path(temporary_directory).glob("*.tmp")))

    def test_csv_outputs_neutralize_formula_cells_but_jsonl_stays_raw(self):
        dangerous_values = (
            "=1+1",
            "+SUM(A1:A2)",
            "-2+3",
            "@SUM(A1:A2)",
            "  =1+1",
            " \ufeff=1+1",
            "\t=1+1",
            "\r+1",
            "\n-1",
        )
        for value in dangerous_values:
            with self.subTest(value=value):
                self.assertEqual(self.evaluator.csv_safe_value(value), "'" + value)
        self.assertEqual(self.evaluator.csv_safe_value("safe text"), "safe text")
        self.assertEqual(self.evaluator.csv_safe_value(-1), -1)
        self.assertEqual(self.evaluator.csv_safe_value(None), "")
        self.assertEqual(
            self.evaluator.csv_safe_value({"text": "=1+1"}),
            '{"text": "=1+1"}',
        )

        report = self.evaluator.evaluate_reasoning(
            [
                {
                    "instance_id": "0",
                    "sample_id": "=HYPERLINK(\"https://example.invalid\")",
                    "target": "safe target",
                    "prediction": "\t@SUM(A1:A2)",
                    "=untrusted_header": "+untrusted value",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.evaluator.write_evaluation_report(
                report, temporary_directory
            )
            jsonl_record = json.loads(
                Path(paths["jsonl"]).read_text(encoding="utf-8").splitlines()[0]
            )
            with Path(paths["csv"]).open(
                encoding="utf-8", newline=""
            ) as csv_file:
                csv_record = next(csv.DictReader(csv_file))

            self.assertEqual(
                jsonl_record["sample_id"],
                "=HYPERLINK(\"https://example.invalid\")",
            )
            self.assertEqual(jsonl_record["prediction"], "\t@SUM(A1:A2)")
            self.assertEqual(
                csv_record["sample_id"],
                "'=HYPERLINK(\"https://example.invalid\")",
            )
            self.assertEqual(csv_record["prediction"], "'\t@SUM(A1:A2)")
            self.assertEqual(csv_record["'=untrusted_header"], "'+untrusted value")

    def test_legacy_reasoning_csv_neutralizes_identifiers_and_predictions(self):
        writer = load_legacy_reasoning_csv_writer(
            self.evaluator.csv_safe_value
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = writer(
                [
                    {
                        "sample_id": "+SUM(A1:A2)",
                        "prediction": "@SUM(A1:A2)",
                    }
                ],
                temporary_directory,
            )
            with Path(path).open(encoding="utf-8", newline="") as csv_file:
                row = next(csv.DictReader(csv_file))

            self.assertEqual(row["names"], "'+SUM(A1:A2)")
            self.assertEqual(row["chi_reasons"], "'@SUM(A1:A2)")


class ValidationTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracker_module = load_source_module("validation_tracker_under_test", TRACKER_PATH)

    def test_first_zero_metric_is_best_and_patience_stops_after_bad_epochs(self):
        tracker = self.tracker_module.ValidationTracker(
            metric_name="macro_f1", patience=2
        )

        self.assertEqual(
            tracker.update(0.0, 0), {"improved": True, "should_stop": False}
        )
        self.assertFalse(tracker.update(0.0, 1)["should_stop"])
        self.assertTrue(tracker.update(-0.1, 2)["should_stop"])
        self.assertEqual(tracker.best_epoch, 0)

    def test_lower_is_better_min_delta_and_state_restore(self):
        tracker = self.tracker_module.ValidationTracker(
            metric_name="loss",
            greater_is_better=False,
            patience=3,
            min_delta=0.1,
        )
        tracker.update(1.0, 0)
        self.assertFalse(tracker.update(0.95, 1)["improved"])
        self.assertTrue(tracker.update(0.8, 2)["improved"])

        restored = self.tracker_module.ValidationTracker(
            metric_name="loss",
            greater_is_better=False,
            patience=3,
            min_delta=0.1,
        )
        restored.load_state_dict(tracker.state_dict())
        self.assertEqual(restored.best_metric, 0.8)
        self.assertEqual(restored.best_epoch, 2)

    def test_resumed_best_checkpoint_prefers_the_resume_sibling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            old_run = root / "old-run"
            new_run = root / "new-run"
            old_run.mkdir()
            new_run.mkdir()
            resume_path = old_run / "checkpoint_last.pth"
            sibling_best = old_run / "checkpoint_best.pth"
            stored_best = root / "stored-best.pth"
            resume_path.write_bytes(b"last")
            sibling_best.write_bytes(b"best")
            stored_best.write_bytes(b"stored")

            resolved = self.tracker_module.resolve_best_checkpoint_path(
                resume_checkpoint_path=resume_path,
                stored_best_checkpoint_path=stored_best,
                output_dir=new_run,
            )

            self.assertEqual(Path(resolved), sibling_best.resolve())

    def test_resumed_best_checkpoint_can_use_a_stored_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stored_best = root / "checkpoint_best.pth"
            stored_best.write_bytes(b"best")

            resolved = self.tracker_module.resolve_best_checkpoint_path(
                resume_checkpoint_path=root / "missing" / "checkpoint_last.pth",
                stored_best_checkpoint_path=stored_best,
                output_dir=root / "new-run",
            )

            self.assertEqual(Path(resolved), stored_best.resolve())

    def test_invalid_metrics_and_patience_fail_early(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            self.tracker_module.ValidationTracker(patience=0)
        tracker = self.tracker_module.ValidationTracker()
        for value in (None, float("nan"), float("inf"), "0.5"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                tracker.update(value, 0)

    def test_split_validation_never_substitutes_test_for_validation(self):
        validate = self.tracker_module.validate_split_configuration

        self.assertIsNone(
            validate(
                available_splits={"train", "test"},
                train_splits=["train"],
                valid_splits=[],
                test_splits=["test"],
            )
        )
        with self.assertRaisesRegex(ValueError, "validation.*missing"):
            validate(
                available_splits={"train", "test"},
                train_splits=["train"],
                valid_splits=["val"],
                test_splits=["test"],
            )
        with self.assertRaisesRegex(ValueError, "both validation and test"):
            validate(
                available_splits={"train", "val"},
                train_splits=["train"],
                valid_splits=["val"],
                test_splits=["val"],
            )

    def test_evaluate_only_requires_an_explicit_test_split(self):
        with self.assertRaisesRegex(ValueError, "explicitly configured test"):
            self.tracker_module.validate_split_configuration(
                available_splits={"train"},
                train_splits=[],
                valid_splits=[],
                test_splits=[],
                evaluate_only=True,
            )

    def test_split_validation_selects_explicit_primary_validation_split(self):
        primary = self.tracker_module.validate_split_configuration(
            available_splits={"train", "dev", "holdout"},
            train_splits=["train"],
            valid_splits=["dev", "holdout"],
            test_splits=[],
            best_model_split="holdout",
        )

        self.assertEqual(primary, "holdout")

    def test_eval_dataset_collapse_rejects_multiple_metric_sources(self):
        collapse = self.tracker_module.collapse_single_eval_datasets
        train_dataset = object()
        val_dataset = object()
        datasets, batch_sizes = collapse(
            {"train": [train_dataset], "val": [val_dataset]},
            {"train": [1], "val": [2]},
            train_splits=["train"],
        )

        self.assertEqual(datasets["train"], [train_dataset])
        self.assertIs(datasets["val"], val_dataset)
        self.assertEqual(batch_sizes["val"], 2)
        with self.assertRaisesRegex(ValueError, "exactly one dataset"):
            collapse(
                {"train": [train_dataset], "val": [val_dataset, object()]},
                {"train": [1], "val": [2, 2]},
                train_splits=["train"],
            )


if __name__ == "__main__":
    unittest.main()
