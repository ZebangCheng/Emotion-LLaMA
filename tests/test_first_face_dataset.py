import ast
from collections.abc import Sequence
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_MODULE_PATH = (
    REPOSITORY_ROOT / "minigpt4" / "datasets" / "datasets" / "first_face.py"
)


class StubVideoCapture:
    def __init__(self, video_path):
        self.video_path = video_path

    def read(self):
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self):
        pass


class ConfigSequence(Sequence):
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


def identity(value):
    return value


def load_dataset_module():
    cv2_was_present = "cv2" in sys.modules
    original_cv2 = sys.modules.get("cv2")
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLOR_BGR2RGB = 0
    cv2_stub.VideoCapture = StubVideoCapture
    cv2_stub.cvtColor = lambda frame, conversion: frame
    sys.modules["cv2"] = cv2_stub

    try:
        spec = importlib.util.spec_from_file_location(
            "first_face_under_test", DATASET_MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if cv2_was_present:
            sys.modules["cv2"] = original_cv2
        else:
            sys.modules.pop("cv2", None)


class DatasetModuleLoaderTest(unittest.TestCase):
    def test_load_dataset_module_restores_cv2_module_state(self):
        cv2_was_present = "cv2" in sys.modules
        original_cv2 = sys.modules.get("cv2")

        try:
            sys.modules.pop("cv2", None)
            load_dataset_module()
            self.assertNotIn("cv2", sys.modules)

            existing_cv2 = types.ModuleType("cv2")
            sys.modules["cv2"] = existing_cv2
            load_dataset_module()
            self.assertIs(sys.modules["cv2"], existing_cv2)
        finally:
            if cv2_was_present:
                sys.modules["cv2"] = original_cv2
            else:
                sys.modules.pop("cv2", None)


class FeatureFaceDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_module = load_dataset_module()
        cls.dataset_cls = cls.dataset_module.FeatureFaceDataset

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = self.temporary_directory.name

        self.ann_path = os.path.join(self.root, "annotations.txt")
        Path(self.ann_path).write_text("sample ignored neutral\n", encoding="utf-8")
        Path(self.root, "coarse.json").write_text(
            '{"sample": {"caption": "coarse"}}', encoding="utf-8"
        )
        Path(self.root, "fine.json").write_text(
            '{"sample": {"smp_reason_caption": "fine"}}', encoding="utf-8"
        )
        Path(self.root, "transcript.csv").write_text(
            "name,sentence\nsample,hello\n", encoding="utf-8"
        )

        for directory, value in (("face", 1.0), ("video", 2.0), ("audio", 3.0)):
            feature_root = Path(self.root, directory)
            feature_root.mkdir()
            np.save(feature_root / "sample.npy", np.array([[value]]))

    def make_dataset(self, **kwargs):
        vis_processor = kwargs.pop("vis_processor", None)
        text_processor = kwargs.pop("text_processor", None)
        annotation_text = kwargs.pop("annotation_text", None)
        if annotation_text is not None:
            Path(self.ann_path).write_text(annotation_text, encoding="utf-8")
        return self.dataset_cls(
            vis_processor,
            text_processor,
            self.root,
            self.ann_path,
            **kwargs,
        )

    def dataset_path_kwargs(self):
        return {
            "coarse_grained_json_path": "coarse.json",
            "fine_grained_json_path": "fine.json",
            "transcription_path": "transcript.csv",
            "face_feature_path": "face",
            "video_feature_path": "video",
            "audio_feature_path": "audio",
        }

    def test_relative_resource_paths_resolve_from_annotation_parent(self):
        dataset = self.make_dataset(**self.dataset_path_kwargs())

        self.assertEqual(dataset.face_feature_path, os.path.join(self.root, "face"))
        self.assertEqual(dataset.video_feature_path, os.path.join(self.root, "video"))
        self.assertEqual(dataset.audio_feature_path, os.path.join(self.root, "audio"))

    def test_path_resolution_accepts_absolute_and_pathlike_values(self):
        absolute_path = Path(self.root, "face")

        self.assertEqual(
            self.dataset_module._resolve_dataset_path(absolute_path, self.root),
            os.path.normpath(os.fspath(absolute_path)),
        )
        self.assertEqual(
            self.dataset_module._resolve_dataset_path(Path("face"), self.root),
            os.path.normpath(os.path.join(self.root, "face")),
        )

    def test_auto_accepts_compact_ne_and_legacy_ncev(self):
        compact = self.make_dataset(annotation_text="sample angry\n")
        legacy = self.make_dataset(annotation_text="sample 35 angry -1.0\n")
        extended_legacy = self.make_dataset(
            annotation_text="sample 35 angry -1.0 retained\n"
        )

        self.assertEqual(compact.samples[0], ("sample", "angry"))
        self.assertEqual(legacy.samples[0], ("sample", "angry"))
        self.assertEqual(extended_legacy.samples[0], ("sample", "angry"))

    def test_explicit_formats_accept_matching_rows(self):
        cases = (
            ("sample angry\n", "ne"),
            ("sample 35 angry\n", "ncev"),
            ("sample 35 angry -1.0\n", "ncev"),
        )
        for annotation_text, annotation_format in cases:
            with self.subTest(annotation_format=annotation_format):
                dataset = self.make_dataset(
                    annotation_text=annotation_text,
                    annotation_format=annotation_format,
                )
                self.assertEqual(dataset.samples, [("sample", "angry")])

    def test_explicit_formats_reject_incompatible_rows(self):
        cases = (
            ("sample angry\n", "ncev", "Invalid NCEV annotation at line 1"),
            (
                "sample 35 angry -1.0 extra\n",
                "ncev",
                "Invalid NCEV annotation at line 1",
            ),
            ("sample 35 angry -1.0\n", "ne", "Invalid NE annotation at line 1"),
        )
        for annotation_text, annotation_format, message in cases:
            with self.subTest(
                annotation_format=annotation_format
            ), self.assertRaisesRegex(ValueError, message):
                self.make_dataset(
                    annotation_text=annotation_text,
                    annotation_format=annotation_format,
                )

    def test_blank_lines_are_ignored_and_repeated_whitespace_is_supported(self):
        dataset = self.make_dataset(
            annotation_text="\n  sample\t\tangry   \n   \n",
        )

        self.assertEqual(dataset.samples, [("sample", "angry")])

    def test_unknown_annotation_format_is_rejected(self):
        for annotation_text in ("sample angry\n", "\n   \n"):
            with self.subTest(annotation_text=annotation_text), self.assertRaisesRegex(
                ValueError, "annotation_format must be one of"
            ):
                self.make_dataset(
                    annotation_text=annotation_text,
                    annotation_format="compact",
                )

    def test_malformed_annotation_rows_include_line_number(self):
        with self.assertRaisesRegex(ValueError, "Invalid NCEV annotation at line 2"):
            self.make_dataset(annotation_text="\nsample\n")

    def test_unknown_emotion_is_rejected_during_initialization(self):
        with self.assertRaisesRegex(ValueError, "Unknown emotion.*ecstatic.*line 1"):
            self.make_dataset(annotation_text="sample ecstatic\n")

    def test_compact_annotation_drives_emotion_item(self):
        dataset = self.make_dataset(
            annotation_text="sample angry\n",
            vis_processor=identity,
            text_processor=identity,
            task_pool=["emotion"],
            transcription_path=None,
            face_feature_path="face",
            video_feature_path="video",
            audio_feature_path="audio",
        )

        sample = dataset[0]

        self.assertEqual(sample["answer"], "angry")
        self.assertEqual(sample["emotion"], dataset.emo2idx["angry"])

    def test_get_uses_configured_feature_roots_in_model_order(self):
        dataset = self.make_dataset(**self.dataset_path_kwargs())

        face, video, audio = dataset.get("sample")

        np.testing.assert_array_equal(face.numpy(), [[1.0]])
        np.testing.assert_array_equal(video.numpy(), [[2.0]])
        np.testing.assert_array_equal(audio.numpy(), [[3.0]])

    def test_emotion_only_does_not_open_reasoning_json_or_transcript(self):
        dataset = self.make_dataset(
            task_pool=["emotion"],
            coarse_grained_json_path="missing-coarse.json",
            fine_grained_json_path="missing-fine.json",
            transcription_path=None,
        )

        self.assertEqual(dataset.task_pool, ["emotion"])
        self.assertIsNone(dataset.MERR_coarse_grained_dict)
        self.assertIsNone(dataset.MERR_fine_grained_dict)
        self.assertIsNone(dataset.character_lines)

    def test_reason_requires_only_coarse_json(self):
        dataset = self.make_dataset(
            task_pool=["reason"],
            coarse_grained_json_path="coarse.json",
            fine_grained_json_path="missing-fine.json",
            transcription_path=None,
        )

        self.assertEqual(
            dataset.MERR_coarse_grained_dict["sample"]["caption"], "coarse"
        )
        self.assertIsNone(dataset.MERR_fine_grained_dict)

    def test_reason_v2_requires_only_fine_json(self):
        dataset = self.make_dataset(
            task_pool=["reason_v2"],
            coarse_grained_json_path="missing-coarse.json",
            fine_grained_json_path="fine.json",
            transcription_path=None,
        )

        self.assertIsNone(dataset.MERR_coarse_grained_dict)
        self.assertEqual(
            dataset.MERR_fine_grained_dict["sample"]["smp_reason_caption"],
            "fine",
        )

    def test_mixed_tasks_load_both_reasoning_files(self):
        dataset = self.make_dataset(
            task_pool=["reason", "emotion", "reason_v2"],
            coarse_grained_json_path="coarse.json",
            fine_grained_json_path="fine.json",
            transcription_path=None,
        )

        self.assertEqual(dataset.task_pool, ["reason", "emotion", "reason_v2"])
        self.assertEqual(
            dataset.MERR_coarse_grained_dict["sample"]["caption"], "coarse"
        )
        self.assertEqual(
            dataset.MERR_fine_grained_dict["sample"]["smp_reason_caption"],
            "fine",
        )

    def test_task_order_and_duplicates_are_preserved_in_a_copy(self):
        task_pool = ["reason_v2", "emotion", "reason_v2"]

        dataset = self.make_dataset(
            task_pool=task_pool,
            coarse_grained_json_path="missing-coarse.json",
            fine_grained_json_path="fine.json",
            transcription_path=None,
        )

        self.assertEqual(dataset.task_pool, task_pool)
        self.assertIsNot(dataset.task_pool, task_pool)

    def test_invalid_task_pools_report_the_complete_contract(self):
        for value in ([], (), "emotion", 7, ["unknown"]):
            with self.subTest(value=value), self.assertRaises(ValueError) as context:
                self.make_dataset(task_pool=value, transcription_path=None)

            message = str(context.exception)
            self.assertIn(repr(value), message)
            self.assertIn("non-empty list or tuple", message)
            for supported_task in ("emotion", "reason", "reason_v2"):
                self.assertIn(supported_task, message)

    def test_enabled_reasoning_task_requires_its_configured_json_path(self):
        cases = (
            (["reason"], "coarse_grained_json_path"),
            (["reason_v2"], "fine_grained_json_path"),
        )
        for task_pool, required_path in cases:
            with self.subTest(task_pool=task_pool), self.assertRaisesRegex(
                ValueError, required_path
            ):
                self.make_dataset(
                    task_pool=task_pool,
                    coarse_grained_json_path=(
                        None
                        if required_path == "coarse_grained_json_path"
                        else "coarse.json"
                    ),
                    fine_grained_json_path=(
                        None
                        if required_path == "fine_grained_json_path"
                        else "fine.json"
                    ),
                    transcription_path=None,
                )

    def test_helper_does_not_hide_missing_reasoning_paths(self):
        cases = (
            (["reason"], "coarse_grained_json_path"),
            (["reason_v2"], "fine_grained_json_path"),
        )
        for task_pool, required_path in cases:
            dataset_kwargs = self.dataset_module.feature_face_dataset_kwargs(
                {"task_pool": task_pool}, {}
            )
            with self.subTest(task_pool=task_pool), self.assertRaisesRegex(
                ValueError, required_path
            ):
                self.make_dataset(**dataset_kwargs)

    def test_optional_transcript_omits_spoken_text_prefix(self):
        dataset = self.make_dataset(
            vis_processor=identity,
            text_processor=identity,
            task_pool=["emotion"],
            coarse_grained_json_path="missing-coarse.json",
            fine_grained_json_path="missing-fine.json",
            transcription_path=None,
            face_feature_path="face",
            video_feature_path="video",
            audio_feature_path="audio",
        )

        sample = dataset[0]

        self.assertNotIn("The person in video says:", sample["instruction_input"])
        self.assertIn("<video><VideoHere></video>", sample["instruction_input"])
        self.assertIn("<feature><FeatureHere></feature>", sample["instruction_input"])
        self.assertIn("[emotion]", sample["instruction_input"])
        self.assertEqual(
            set(sample),
            {
                "image",
                "video_features",
                "instruction_input",
                "answer",
                "emotion",
                "image_id",
            },
        )

    def test_evaluation_mode_adds_stable_metadata_and_preserves_raw_target(self):
        def normalized_text(value):
            return value.lower().replace("!", "")

        Path(self.root, "fine.json").write_text(
            '{"sample": {"smp_reason_caption": "Raised Voice!"}}',
            encoding="utf-8",
        )
        dataset = self.make_dataset(
            vis_processor=identity,
            text_processor=normalized_text,
            task_pool=["reason_v2"],
            fine_grained_json_path="fine.json",
            transcription_path=None,
            face_feature_path="face",
            video_feature_path="video",
            audio_feature_path="audio",
            evaluation_mode=True,
            split="val",
        )
        dataset.name = "feature_face_caption"

        first = dataset[0]
        second = dataset[0]

        self.assertEqual(first["task"], "reason_v2")
        self.assertEqual(first["target_raw"], "Raised Voice!")
        self.assertEqual(first["answer"], "raised voice")
        self.assertEqual(first["sample_index"], 0)
        self.assertEqual(first["sample_id"], "sample")
        self.assertEqual(first["split"], "val")
        self.assertEqual(first["instance_id"], second["instance_id"])
        self.assertEqual(first["instruction_input"], second["instruction_input"])

    def test_evaluation_mode_requires_one_task(self):
        with self.assertRaisesRegex(ValueError, "exactly one task"):
            self.make_dataset(
                task_pool=["emotion", "reason"],
                coarse_grained_json_path="coarse.json",
                evaluation_mode=True,
            )

    def test_configured_labels_drive_targets_and_instruction(self):
        dataset = self.make_dataset(
            annotation_text="sample calm\n",
            vis_processor=identity,
            text_processor=identity,
            task_pool=["emotion"],
            labels=["calm", "excited"],
            transcription_path=None,
            face_feature_path="face",
            video_feature_path="video",
            audio_feature_path="audio",
            evaluation_mode=True,
        )

        sample = dataset[0]

        self.assertEqual(dataset.labels, ["calm", "excited"])
        self.assertEqual(sample["answer"], "calm")
        self.assertIn("calm, excited", sample["instruction_input"])

    def test_transcript_requires_name_and_sentence_columns(self):
        invalid_transcripts = {
            "missing-name.csv": "speaker,sentence\nsample,hello\n",
            "missing-sentence.csv": "name,text\nsample,hello\n",
        }
        for filename, contents in invalid_transcripts.items():
            Path(self.root, filename).write_text(contents, encoding="utf-8")
            missing_column = "name" if "missing-name" in filename else "sentence"
            with self.subTest(filename=filename), self.assertRaisesRegex(
                ValueError, missing_column
            ):
                self.make_dataset(
                    task_pool=["emotion"],
                    transcription_path=filename,
                    coarse_grained_json_path="missing-coarse.json",
                    fine_grained_json_path="missing-fine.json",
                )

    def test_transcript_requires_a_row_for_the_current_sample(self):
        Path(self.root, "other-transcript.csv").write_text(
            "name,sentence\nother,hello\n", encoding="utf-8"
        )
        dataset = self.make_dataset(
            vis_processor=identity,
            text_processor=identity,
            task_pool=["emotion"],
            transcription_path="other-transcript.csv",
            face_feature_path="face",
            video_feature_path="video",
            audio_feature_path="audio",
        )

        with self.assertRaisesRegex(KeyError, "sample"):
            dataset[0]

    def test_helper_preserves_invalid_task_pool_for_focused_validation(self):
        for task_pool in ("emotion", 7):
            with self.subTest(task_pool=task_pool), self.assertRaisesRegex(
                ValueError, "Invalid task_pool"
            ):
                dataset_kwargs = self.dataset_module.feature_face_dataset_kwargs(
                    {"task_pool": task_pool}, {}
                )
                self.make_dataset(**dataset_kwargs)

    def test_helper_preserves_config_sequence_until_constructor_validation(self):
        task_pool = ConfigSequence(["emotion"])

        dataset_kwargs = self.dataset_module.feature_face_dataset_kwargs(
            {"task_pool": task_pool}, {}
        )
        self.assertIs(dataset_kwargs["task_pool"], task_pool)

        dataset = self.make_dataset(**dataset_kwargs)
        self.assertEqual(dataset.task_pool, ["emotion"])

    def test_dataset_kwargs_forward_task_pool_from_behavioral_config(self):
        task_pool = ["reason_v2", "emotion"]

        kwargs = self.dataset_module.feature_face_dataset_kwargs(
            {"task_pool": task_pool},
            {"face_feature_path": "face"},
        )

        self.assertEqual(kwargs["task_pool"], task_pool)
        self.assertEqual(kwargs["face_feature_path"], "face")

    def test_dataset_kwargs_forward_exact_behavior_and_path_contract(self):
        task_pool = ["emotion", "reason"]
        behavioral_config = {
            "task_pool": task_pool,
            "annotation_format": "auto",
            "ignored_behavior": "not-forwarded",
        }
        path_config = {
            key: Path("configured", key) for key in self.dataset_module.FEATURE_FACE_PATH_KEYS
        }
        path_config["ignored_path"] = "not-forwarded"

        kwargs = self.dataset_module.feature_face_dataset_kwargs(
            behavioral_config,
            path_config,
        )

        self.assertEqual(
            set(kwargs),
            {"task_pool", "annotation_format", *self.dataset_module.FEATURE_FACE_PATH_KEYS},
        )
        self.assertIs(kwargs["task_pool"], task_pool)
        self.assertEqual(kwargs["annotation_format"], "auto")
        for key in self.dataset_module.FEATURE_FACE_PATH_KEYS:
            self.assertIs(kwargs[key], path_config[key])

    def test_dataset_kwargs_forward_annotation_format(self):
        kwargs = self.dataset_module.feature_face_dataset_kwargs(
            {"annotation_format": "ne"},
            {},
        )

        self.assertEqual(kwargs["annotation_format"], "ne")

    def test_dataset_kwargs_forward_configured_labels(self):
        labels = ["calm", "excited"]
        kwargs = self.dataset_module.feature_face_dataset_kwargs(
            {"labels": labels},
            {},
        )

        self.assertIs(kwargs["labels"], labels)


class FeatureFaceConfigForwardingTest(unittest.TestCase):
    def test_builder_forwards_shared_dataset_kwargs_to_dataset_constructor(self):
        source = (
            REPOSITORY_ROOT
            / "minigpt4"
            / "datasets"
            / "builders"
            / "image_text_pair_builder.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        helper_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "feature_face_dataset_kwargs"
        ]
        self.assertTrue(
            any(
                len(call.args) == 2
                and isinstance(call.args[0], ast.Attribute)
                and isinstance(call.args[0].value, ast.Name)
                and call.args[0].value.id == "self"
                and call.args[0].attr == "config"
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id == "build_info"
                for call in helper_calls
            )
        )

        dataset_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dataset_cls"
        ]
        self.assertTrue(
            any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "dataset_kwargs"
                for call in dataset_calls
                for keyword in call.keywords
            )
        )

    def test_evaluation_scripts_delegate_to_shared_cli(self):
        for relative_path in ("eval_emotion.py", "eval_emotion_EMER.py"):
            with self.subTest(relative_path=relative_path):
                source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
                tree = ast.parse(source)
                shared_imports = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "minigpt4.evaluation.cli"
                    and any(alias.name == "main" for alias in node.names)
                ]
                main_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "main"
                ]

                self.assertTrue(shared_imports)
                self.assertTrue(main_calls)

    def test_shared_evaluation_cli_uses_dataset_config_forwarder(self):
        source = (
            REPOSITORY_ROOT / "minigpt4" / "evaluation" / "cli.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "feature_face_dataset_kwargs"
        ]
        dataset_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FeatureFaceDataset"
        ]

        self.assertTrue(helper_calls)
        self.assertTrue(
            any(
                keyword.arg is None
                for call in dataset_calls
                for keyword in call.keywords
            )
        )

    def test_shipped_configs_expose_dataset_behavior(self):
        def load_config(relative_path):
            return yaml.safe_load(
                (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
            )

        path_keys = {
            "transcription_path",
            "face_feature_path",
            "video_feature_path",
            "audio_feature_path",
        }
        default_dataset = load_config(
            "minigpt4/configs/datasets/firstface/featureface.yaml"
        )["datasets"]["feature_face_caption"]
        self.assertEqual(default_dataset["task_pool"], ["emotion"])
        self.assertEqual(default_dataset["annotation_format"], "auto")
        self.assertTrue(
            path_keys
            | {"image_path", "ann_path", "coarse_grained_json_path"}
            <= set(default_dataset["build_info"])
        )

        training_configs = (
            (
                "train_configs/Emotion-LLaMA_finetune.yaml",
                ["emotion", "reason"],
                "coarse_grained_json_path",
            ),
            (
                "train_configs/minigptv2_tuning_stage_2.yaml",
                ["reason_v2"],
                "fine_grained_json_path",
            ),
        )
        for relative_path, task_pool, reasoning_path_key in training_configs:
            with self.subTest(relative_path=relative_path):
                dataset = load_config(relative_path)["datasets"][
                    "feature_face_caption"
                ]
                self.assertEqual(dataset["task_pool"], task_pool)
                self.assertEqual(dataset["annotation_format"], "auto")
                self.assertNotIn("task_pool", dataset["build_info"])
                self.assertNotIn("annotation_format", dataset["build_info"])
                self.assertTrue(
                    path_keys | {"image_path", "ann_path", reasoning_path_key}
                    <= set(dataset["build_info"])
                )

        evaluation_configs = (
            ("eval_configs/eval_emotion.yaml", ["emotion"], None),
            (
                "eval_configs/eval_emotion_EMER.yaml",
                ["reason_v2"],
                "fine_grained_json_path",
            ),
        )
        for relative_path, task_pool, reasoning_path_key in evaluation_configs:
            with self.subTest(relative_path=relative_path):
                dataset = load_config(relative_path)["evaluation_datasets"][
                    "feature_face_caption"
                ]
                self.assertEqual(dataset["task_pool"], task_pool)
                self.assertEqual(dataset["annotation_format"], "auto")
                self.assertTrue(
                    path_keys | {"eval_file_path", "img_path"} <= set(dataset)
                )
                self.assertNotIn("build_info", dataset)
                if relative_path.endswith("eval_emotion.yaml"):
                    self.assertEqual(
                        dataset["labels"],
                        ["neutral", "angry", "happy", "sad", "worried", "surprise"],
                    )
                if reasoning_path_key is not None:
                    self.assertIn(reasoning_path_key, dataset)
                    self.assertTrue(
                        dataset["eval_file_path"].endswith("MERR_fine_grained.txt")
                    )


if __name__ == "__main__":
    unittest.main()
