import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_MODULE_PATH = (
    REPOSITORY_ROOT / "minigpt4" / "datasets" / "datasets" / "first_face.py"
)


def load_dataset_module():
    cv2_was_present = "cv2" in sys.modules
    original_cv2 = sys.modules.get("cv2")
    cv2_stub = types.ModuleType("cv2")
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
        Path(self.root, "coarse.json").write_text("{}", encoding="utf-8")
        Path(self.root, "fine.json").write_text("{}", encoding="utf-8")
        Path(self.root, "transcript.csv").write_text(
            "name,sentence\nsample,hello\n", encoding="utf-8"
        )

        for directory, value in (("face", 1.0), ("video", 2.0), ("audio", 3.0)):
            feature_root = Path(self.root, directory)
            feature_root.mkdir()
            np.save(feature_root / "sample.npy", np.array([[value]]))

    def make_dataset(self, **kwargs):
        return self.dataset_cls(
            None,
            None,
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

    def test_get_uses_configured_feature_roots_in_model_order(self):
        dataset = self.make_dataset(**self.dataset_path_kwargs())

        face, video, audio = dataset.get("sample")

        np.testing.assert_array_equal(face.numpy(), [[1.0]])
        np.testing.assert_array_equal(video.numpy(), [[2.0]])
        np.testing.assert_array_equal(audio.numpy(), [[3.0]])

if __name__ == "__main__":
    unittest.main()
