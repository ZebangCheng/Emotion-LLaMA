import importlib.util
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).parents[1]


def _load_conversation_module():
    """Load the real module without requiring its optional ML dependencies."""
    torch = types.ModuleType("torch")
    torch.LongTensor = object
    torch.FloatTensor = object
    torch.Tensor = object

    transformers = types.ModuleType("transformers")

    class StoppingCriteria:
        pass

    transformers.StoppingCriteria = StoppingCriteria
    transformers.StoppingCriteriaList = list
    for name in (
        "AutoTokenizer",
        "AutoModelForCausalLM",
        "LlamaTokenizer",
        "TextIteratorStreamer",
        "Wav2Vec2FeatureExtractor",
    ):
        setattr(transformers, name, object)

    moviepy = types.ModuleType("moviepy")
    moviepy_editor = types.ModuleType("moviepy.editor")
    moviepy_editor.VideoFileClip = object
    moviepy.editor = moviepy_editor

    minigpt4 = types.ModuleType("minigpt4")
    minigpt4.__path__ = []
    common = types.ModuleType("minigpt4.common")
    common.__path__ = []
    registry = types.ModuleType("minigpt4.common.registry")
    registry.registry = object()

    soundfile = types.ModuleType("soundfile")
    soundfile.read = lambda _path: None
    stub_modules = {
        "torch": torch,
        "transformers": transformers,
        "cv2": types.ModuleType("cv2"),
        "moviepy": moviepy,
        "moviepy.editor": moviepy_editor,
        "soundfile": soundfile,
        "minigpt4": minigpt4,
        "minigpt4.common": common,
        "minigpt4.common.registry": registry,
    }

    module_path = REPO_ROOT / "minigpt4" / "conversation" / "conversation.py"
    module_name = f"conversation_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stub_modules):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


class _FakeAudio:
    def __init__(self, written_paths):
        self.written_paths = written_paths

    def write_audiofile(self, path, **_kwargs):
        self.written_paths.append(path)
        Path(path).write_bytes(b"fake wav")


class _FakeVideoClip:
    def __init__(self, written_paths):
        self.audio = _FakeAudio(written_paths)
        self.closed = False

    def close(self):
        self.closed = True


class ExtractAudioFromVideoTests(unittest.TestCase):
    def setUp(self):
        self.conversation = _load_conversation_module()
        self.written_paths = []
        self.clips = []

    def open_video(self, _video_path):
        clip = _FakeVideoClip(self.written_paths)
        self.clips.append(clip)
        return clip

    def test_uses_unique_temporary_files_and_cleans_them(self):
        read_paths = []

        def read_audio(path):
            path = Path(path).resolve()
            self.assertTrue(path.exists())
            read_paths.append(path)
            return [0.1, -0.1], 16000

        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as working_dir:
            previous_dir = os.getcwd()
            os.chdir(working_dir)
            try:
                with mock.patch.object(
                    self.conversation, "VideoFileClip", self.open_video
                ), mock.patch.object(self.conversation.sf, "read", read_audio):
                    first = self.conversation.extract_audio_from_video("first.mp4")
                    second = self.conversation.extract_audio_from_video("second.mp4")
            finally:
                os.chdir(previous_dir)

            self.assertFalse((Path(working_dir) / "audio.wav").exists())

        self.assertEqual(first, ([0.1, -0.1], 16000))
        self.assertEqual(second, ([0.1, -0.1], 16000))
        self.assertEqual(
            [Path(path).resolve() for path in self.written_paths], read_paths
        )
        self.assertNotEqual(self.written_paths[0], self.written_paths[1])
        self.assertTrue(all(not Path(path).exists() for path in self.written_paths))
        self.assertTrue(all(clip.closed for clip in self.clips))

    def test_cleans_resources_when_reading_fails(self):
        def fail_to_read(_path):
            raise RuntimeError("cannot read wav")

        with mock.patch.object(
            self.conversation, "VideoFileClip", self.open_video
        ), mock.patch.object(self.conversation.sf, "read", fail_to_read):
            with self.assertRaisesRegex(RuntimeError, "cannot read wav"):
                self.conversation.extract_audio_from_video("broken.mp4")

        self.assertEqual(len(self.written_paths), 1)
        self.assertFalse(Path(self.written_paths[0]).exists())
        self.assertTrue(self.clips[0].closed)


if __name__ == "__main__":
    unittest.main()
