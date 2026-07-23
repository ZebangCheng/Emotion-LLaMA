import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    REPOSITORY_ROOT / "minigpt4" / "datasets" / "datasets" / "mer2024.py"
)


def load_dataset_module():
    original_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = types.SimpleNamespace()
    try:
        spec = importlib.util.spec_from_file_location(
            "mer2024_dataset_under_test", DATASET_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if original_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = original_cv2


class MER2024DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_dataset_module()

    def make_feature_tree(self, root, relative_path, value):
        feature_root = root / relative_path
        feature_root.mkdir(parents=True)
        np.save(feature_root / "sample.npy", np.array([[value, value]]))

    def test_configured_resources_labels_and_evaluation_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "annotations.txt"
            annotation_path.write_text("sample 12 fear 0.0\n", encoding="utf-8")
            (root / "transcripts.csv").write_text(
                "name,sentence_en\nsample,hello world\n", encoding="utf-8"
            )
            self.make_feature_tree(root, "features/face", 1)
            self.make_feature_tree(root, "features/video", 2)
            self.make_feature_tree(root, "features/audio", 3)

            dataset = self.module.MER2024Dataset(
                vis_processor=lambda image: torch.zeros(3, 2, 2),
                text_processor=lambda text: "processed:{}".format(text),
                vis_root=root / "videos",
                ann_path=annotation_path,
                labels=["neutral", "fear"],
                transcription_path="transcripts.csv",
                face_feature_path="features/face",
                video_feature_path="features/video",
                audio_feature_path="features/audio",
                evaluation_mode=True,
                split="val",
            )
            dataset.name = "mer2024_caption"
            dataset.extract_frame = lambda path: np.zeros((2, 2, 3), dtype=np.uint8)

            sample = dataset[0]

            self.assertIn("neutral, fear", sample["instruction_input"])
            self.assertIn("hello world", sample["instruction_input"])
            self.assertEqual(sample["answer"], "processed:fear")
            self.assertEqual(sample["target_raw"], "fear")
            self.assertEqual(sample["split"], "val")
            self.assertEqual(sample["instance_id"], "mer2024_caption:val:0:emotion:sample")
            self.assertTrue(
                torch.equal(
                    sample["video_features"],
                    torch.tensor([[1, 1], [2, 2], [3, 3]]),
                )
            )

    def test_transcript_is_optional_and_compact_annotations_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_path = Path(temporary_directory) / "annotations.txt"
            annotation_path.write_text("sample happy\n", encoding="utf-8")

            dataset = self.module.MER2024Dataset(
                vis_processor=lambda image: image,
                text_processor=lambda text: text,
                vis_root=temporary_directory,
                ann_path=annotation_path,
                labels=["happy"],
            )

            self.assertEqual(len(dataset), 1)
            self.assertIsNone(dataset.character_lines)
            self.assertEqual(dataset.labels, ["happy"])

    def test_config_forwarder_and_unknown_labels_fail_closed(self):
        kwargs = self.module.mer2024_dataset_kwargs(
            {"labels": ["happy"]},
            {
                "transcription_path": "transcripts.csv",
                "face_feature_path": "face",
                "video_feature_path": "video",
                "audio_feature_path": "audio",
            },
        )
        self.assertEqual(
            kwargs,
            {
                "labels": ["happy"],
                "transcription_path": "transcripts.csv",
                "face_feature_path": "face",
                "video_feature_path": "video",
                "audio_feature_path": "audio",
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_path = Path(temporary_directory) / "annotations.txt"
            annotation_path.write_text("sample 1 sad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown emotion label"):
                self.module.MER2024Dataset(
                    vis_processor=lambda image: image,
                    text_processor=lambda text: text,
                    vis_root=temporary_directory,
                    ann_path=annotation_path,
                    labels=["happy"],
                )

    def test_builder_and_offline_cli_use_the_shared_forwarder(self):
        builder_source = (
            REPOSITORY_ROOT
            / "minigpt4"
            / "datasets"
            / "builders"
            / "image_text_pair_builder.py"
        ).read_text(encoding="utf-8")
        cli_source = (
            REPOSITORY_ROOT / "minigpt4" / "evaluation" / "cli.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "mer2024_dataset_kwargs(self.config, build_info)", builder_source
        )
        self.assertIn("**mer2024_dataset_kwargs(dataset_cfg)", cli_source)


if __name__ == "__main__":
    unittest.main()
