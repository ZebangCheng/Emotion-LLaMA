import ast
import datetime
import importlib.util
import logging
from pathlib import Path
import sys
import time
import types
import unittest

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = REPOSITORY_ROOT / "minigpt4" / "tasks" / "image_text_pretrain.py"
RUNNER_PATH = REPOSITORY_ROOT / "minigpt4" / "runners" / "runner_base.py"
TRACKER_PATH = REPOSITORY_ROOT / "minigpt4" / "evaluation" / "tracker.py"


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def load_tracker_module():
    spec = importlib.util.spec_from_file_location(
        "validation_tracker_for_runner_test", TRACKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task_module():
    original_modules = {}
    stub_names = (
        "minigpt4.common.dist_utils",
        "minigpt4.common.registry",
        "minigpt4.conversation.conversation",
        "minigpt4.evaluation.distributed",
        "minigpt4.evaluation.evaluator",
        "minigpt4.evaluation.prompting",
        "minigpt4.tasks.base_task",
    )
    for name in stub_names:
        original_modules[name] = sys.modules.get(name)

    class RegistryStub:
        def register_task(self, name):
            return lambda task_class: task_class

        def get_path(self, name):
            return "."

    class BaseTaskStub:
        def __init__(self):
            self.cfg = ""

    conversation = types.SimpleNamespace(copy=lambda: types.SimpleNamespace(system=""))
    stubs = {
        "minigpt4.common.dist_utils": types.SimpleNamespace(
            is_main_process=lambda: False
        ),
        "minigpt4.common.registry": types.SimpleNamespace(registry=RegistryStub()),
        "minigpt4.conversation.conversation": types.SimpleNamespace(
            CONV_VISION_minigptv2=conversation
        ),
        "minigpt4.evaluation.distributed": types.SimpleNamespace(
            gather_evaluation_records=lambda records: list(records)
        ),
        "minigpt4.evaluation.evaluator": types.SimpleNamespace(
            evaluate_records=lambda records, **kwargs: {
                "records": records,
                "metrics": {"agg_metrics": 1.0},
            },
            write_evaluation_report=lambda report, output_dir: None,
        ),
        "minigpt4.evaluation.prompting": types.SimpleNamespace(
            prepare_conversation_texts=lambda texts, template: [
                "prompt:{}".format(text) for text in texts
            ]
        ),
        "minigpt4.tasks.base_task": types.SimpleNamespace(BaseTask=BaseTaskStub),
    }
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "image_text_pretrain_under_test", TASK_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def load_runner_train_function():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RunnerBase"
    )
    train_method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    train_method.decorator_list = []
    module = ast.Module(body=[train_method], type_ignores=[])
    namespace = {
        "datetime": datetime,
        "logging": logging,
        "time": time,
        "dist": types.SimpleNamespace(barrier=lambda: None),
    }
    exec(compile(module, str(RUNNER_PATH), "exec"), namespace)
    return namespace["train"]


class EvaluationTaskRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_module = load_task_module()

    def test_valid_step_generates_structured_records_with_raw_targets(self):
        task = self.task_module.ImageTextPretrainTask()
        task.cfg = types.SimpleNamespace(
            run_cfg={
                "evaluation": {
                    "generation": {"max_new_tokens": 7, "num_beams": 2}
                }
            }
        )

        class FakeModel:
            def __init__(self):
                self.call = None

            def generate(self, images, video_features, texts, **kwargs):
                self.call = {"texts": texts, "kwargs": kwargs}
                return ["The emotion is happy."]

        model = FakeModel()
        records = task.valid_step(
            model,
            {
                "image": torch.zeros(1),
                "video_features": torch.zeros(1),
                "instruction_input": ["instruction"],
                "instance_id": ["dataset:val:0"],
                "sample_id": ["sample"],
                "sample_index": torch.tensor([0]),
                "dataset": ["dataset"],
                "split": ["val"],
                "task": ["emotion"],
                "target_raw": ["happy"],
                "answer": ["processed"],
            },
        )

        self.assertEqual(model.call["texts"], ["prompt:instruction"])
        self.assertEqual(model.call["kwargs"]["max_new_tokens"], 7)
        self.assertEqual(model.call["kwargs"]["num_beams"], 2)
        self.assertFalse(model.call["kwargs"]["do_sample"])
        self.assertEqual(records[0]["target"], "happy")
        self.assertEqual(records[0]["prediction"], "The emotion is happy.")
        self.assertEqual(records[0]["instance_id"], "dataset:val:0")

    def test_valid_step_requires_stable_instance_ids(self):
        task = self.task_module.ImageTextPretrainTask()
        task.cfg = types.SimpleNamespace(run_cfg={"evaluation": {}})

        class FakeModel:
            def generate(self, *args, **kwargs):
                return ["happy"]

        with self.assertRaisesRegex(ValueError, "stable instance_id"):
            task.valid_step(
                FakeModel(),
                {
                    "image": torch.zeros(1),
                    "video_features": torch.zeros(1),
                    "instruction_input": ["instruction"],
                },
            )


class RunnerValidationLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_function = staticmethod(load_runner_train_function())
        cls.tracker_module = load_tracker_module()

    def make_runner(
        self,
        metrics=(),
        valid_splits=("val",),
        test_splits=("test",),
        evaluate_only=False,
        patience=None,
    ):
        runner = types.SimpleNamespace()
        runner.config = types.SimpleNamespace(
            run_cfg=AttrDict(distributed=False)
        )
        runner.start_epoch = 0
        runner.max_epoch = max(len(metrics), 1)
        runner.valid_splits = list(valid_splits)
        runner.test_splits = list(test_splits)
        runner.best_model_split = "val" if valid_splits else None
        runner.evaluate_only = evaluate_only
        runner.resume_ckpt_path = None
        runner.validation_tracker = self.tracker_module.ValidationTracker(
            metric_name="macro_f1", patience=patience
        )
        runner.log_config = lambda: None
        runner._validate_splits = lambda: None
        runner._load_checkpoint = lambda path: None
        runner.train_epoch = lambda epoch: {"loss": "0.1"}
        runner.logged = []
        runner.log_stats = lambda stats, split_name: runner.logged.append(
            (split_name, dict(stats))
        )
        metric_values = iter(metrics)
        runner.eval_epoch = lambda split_name, cur_epoch, **kwargs: {
            "macro_f1": next(metric_values),
            "agg_metrics": 0.0,
        }
        runner.saved = []
        runner._save_checkpoint = lambda epoch, **kwargs: runner.saved.append(
            (epoch, kwargs)
        )
        runner._broadcast_early_stop = lambda value: bool(value)
        runner.evaluated = []
        runner.evaluate = lambda **kwargs: runner.evaluated.append(kwargs)
        return runner

    def test_best_last_and_early_stopping_lifecycle(self):
        runner = self.make_runner(
            metrics=[0.2, 0.3, 0.25, 0.24], patience=2
        )

        self.train_function(runner)

        best_saves = [entry for entry in runner.saved if entry[1].get("is_best")]
        last_saves = [
            entry
            for entry in runner.saved
            if entry[1].get("checkpoint_name") == "last"
        ]
        self.assertEqual([entry[0] for entry in best_saves], [0, 1])
        self.assertEqual([entry[0] for entry in last_saves], [0, 1, 2, 3])
        self.assertEqual(runner.validation_tracker.best_epoch, 1)
        self.assertEqual(runner.validation_tracker.bad_epochs, 2)
        self.assertEqual(
            runner.evaluated,
            [{"cur_epoch": "best", "skip_reload": False}],
        )

    def test_validation_disabled_preserves_numbered_checkpoint_behavior(self):
        runner = self.make_runner(
            metrics=[], valid_splits=(), test_splits=(), patience=None
        )

        self.train_function(runner)

        self.assertEqual(runner.saved, [(0, {"is_best": False})])
        self.assertFalse(runner.evaluated)

    def test_evaluate_only_never_trains_or_saves(self):
        runner = self.make_runner(
            metrics=[],
            valid_splits=(),
            test_splits=("test",),
            evaluate_only=True,
        )
        runner.train_epoch = lambda epoch: self.fail("training must not run")

        self.train_function(runner)

        self.assertFalse(runner.saved)
        self.assertEqual(
            runner.evaluated,
            [{"cur_epoch": "provided", "skip_reload": True}],
        )

    def test_runner_source_persists_tracker_and_passes_cpu_eval_flag(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")

        self.assertIn('"runner_state": self.validation_tracker.state_dict()', source)
        self.assertIn(
            'self.validation_tracker.load_state_dict(checkpoint.get("runner_state"))',
            source,
        )
        self.assertIn("cuda_enabled=self.cuda_enabled", source)


if __name__ == "__main__":
    unittest.main()
