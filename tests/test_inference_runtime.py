import importlib.util
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).parents[1]


def _load_source_module(relative_path, prefix, stub_modules=None):
    module_path = REPO_ROOT / relative_path
    module_name = "{}_{}".format(prefix, uuid.uuid4().hex)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    stubs = stub_modules or {}
    with mock.patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def _load_runtime_module():
    return _load_source_module(
        Path("minigpt4") / "inference" / "runtime.py",
        "runtime_under_test",
    )


def _load_minigpt_v2_module():
    torch = types.ModuleType("torch")
    torch_cuda = types.ModuleType("torch.cuda")
    torch_cuda_amp = types.ModuleType("torch.cuda.amp")
    torch_cuda_amp.autocast = object
    torch_cuda.amp = torch_cuda_amp
    torch.cuda = torch_cuda

    torch_nn = types.ModuleType("torch.nn")
    torch_nn.Module = object
    torch_nn.Linear = object
    torch.nn = torch_nn

    registry_module = types.ModuleType("minigpt4.common.registry")

    class FakeRegistry:
        @staticmethod
        def register_model(_name):
            return lambda model_class: model_class

    registry_module.registry = FakeRegistry()
    base_model = types.ModuleType("minigpt4.models.base_model")
    base_model.disabled_train = lambda *args, **kwargs: None
    minigpt_base = types.ModuleType("minigpt4.models.minigpt_base")
    minigpt_base.MiniGPTBase = object
    qformer = types.ModuleType("minigpt4.models.Qformer")
    qformer.BertConfig = object
    qformer.BertLMHeadModel = object

    return _load_source_module(
        Path("minigpt4") / "models" / "minigpt_v2.py",
        "minigpt_v2_under_test",
        {
            "torch": torch,
            "torch.cuda": torch_cuda,
            "torch.cuda.amp": torch_cuda_amp,
            "torch.nn": torch_nn,
            "minigpt4.common.registry": registry_module,
            "minigpt4.models.base_model": base_model,
            "minigpt4.models.minigpt_base": minigpt_base,
            "minigpt4.models.Qformer": qformer,
        },
    )


class _FakeDevice:
    def __init__(self, value):
        value = str(value)
        if value == "cuda":
            self.type = "cuda"
            self.index = None
        elif value.startswith("cuda:"):
            self.type = "cuda"
            self.index = int(value.split(":", 1)[1])
        else:
            self.type = value
            self.index = None

    def __str__(self):
        if self.index is None:
            return self.type
        return "{}:{}".format(self.type, self.index)


class _FakeCuda:
    def __init__(self, *, available, current_device, device_count):
        self._available = available
        self._current_device = current_device
        self._device_count = device_count

    def is_available(self):
        return self._available

    def current_device(self):
        return self._current_device

    def device_count(self):
        return self._device_count


class _FakeTorch:
    def __init__(self, *, available=True, current_device=0, device_count=1):
        self.cuda = _FakeCuda(
            available=available,
            current_device=current_device,
            device_count=device_count,
        )

    @staticmethod
    def device(value):
        return _FakeDevice(value)


def _load_conversation_module():
    torch = types.ModuleType("torch")
    torch.LongTensor = object
    torch.FloatTensor = object

    class Tensor:
        pass

    torch.Tensor = Tensor

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
    module = _load_source_module(
        Path("minigpt4") / "conversation" / "conversation.py",
        "conversation_under_test",
        stub_modules,
    )
    return module, transformers


class _FakeConversation:
    def __init__(self, sequence):
        self.sequence = sequence
        self.messages = []


class _FakeChat:
    def __init__(self):
        self.conversations = []
        self.image_lists = []

    def upload_img(self, video_path, conversation, image_list):
        self.conversations.append(conversation)
        self.image_lists.append(image_list)
        conversation.messages.append(["video", video_path])
        image_list.append(video_path)

    def ask(self, prompt, conversation):
        conversation.messages.append(["prompt", prompt])

    def encode_img(self, image_list):
        image_list[:] = ["encoded:" + image_list[0]]

    def answer(self, conv, img_list, **_kwargs):
        answer = "answer-{}".format(conv.sequence)
        conv.messages.append(["answer", answer])
        return answer, []


class InferenceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_module = _load_runtime_module()

    def test_loader_runs_once_and_each_request_gets_fresh_state(self):
        loader_calls = []
        conversations = []
        chat = _FakeChat()

        def conversation_factory():
            conversation = _FakeConversation(len(conversations) + 1)
            conversations.append(conversation)
            return conversation

        def component_loader():
            loader_calls.append(True)
            return chat, conversation_factory, "cpu"

        runtime = self.runtime_module.EmotionLLaMARuntime(
            component_loader=component_loader
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = Path(temporary_directory) / "sample.mp4"
            video.write_bytes(b"video")

            self.assertFalse(runtime.is_loaded)
            self.assertIs(runtime.load(), runtime)
            self.assertIs(runtime.load(), runtime)
            first = runtime.analyze(video, "first prompt")
            second = runtime.analyze(video, "second prompt")

        self.assertEqual(loader_calls, [True])
        self.assertTrue(runtime.is_loaded)
        self.assertEqual((first, second), ("answer-1", "answer-2"))
        self.assertEqual(len(conversations), 2)
        self.assertIsNot(conversations[0], conversations[1])
        self.assertIsNot(chat.image_lists[0], chat.image_lists[1])
        self.assertEqual(
            [message[1] for message in conversations[0].messages],
            [str(video.resolve()), "first prompt", "answer-1"],
        )
        self.assertEqual(
            [message[1] for message in conversations[1].messages],
            [str(video.resolve()), "second prompt", "answer-2"],
        )

    def test_invalid_seed_is_rejected_before_component_loading(self):
        loader_calls = []

        def component_loader():
            loader_calls.append(True)
            return _FakeChat(), lambda: _FakeConversation(1), "cpu"

        runtime = self.runtime_module.EmotionLLaMARuntime(
            component_loader=component_loader
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = Path(temporary_directory) / "sample.mp4"
            video.write_bytes(b"video")
            for seed in (-1, True, 1.5, "7"):
                with self.subTest(seed=seed):
                    with self.assertRaisesRegex(
                        ValueError, "seed must be a non-negative integer"
                    ):
                        runtime.analyze(video, "prompt", seed=seed)

        self.assertEqual(loader_calls, [])
        self.assertFalse(runtime.is_loaded)

    def test_documented_relative_model_paths_resolve_from_repo_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            config = {
                "ckpt": "checkpoints/emotion.pth",
                "llama_model": "checkpoints/llama",
            }
            self.runtime_module.EmotionLLaMARuntime._normalize_model_paths(
                config, repo_root
            )

            self.assertEqual(
                config["ckpt"], str((repo_root / "checkpoints/emotion.pth").resolve())
            )
            self.assertEqual(
                config["llama_model"],
                str((repo_root / "checkpoints/llama").resolve()),
            )

            remote_config = {"llama_model": "organization/model-name"}
            self.runtime_module.EmotionLLaMARuntime._normalize_model_paths(
                remote_config, repo_root
            )
            self.assertEqual(
                remote_config["llama_model"], "organization/model-name"
            )

    def test_secondary_cuda_device_is_injected_before_quantized_load(self):
        model_config = {"low_resource": True, "device_8bit": 0}
        device = self.runtime_module.EmotionLLaMARuntime._configure_model_device(
            _FakeTorch(device_count=2), model_config, "cuda:1"
        )

        self.assertEqual(str(device), "cuda:1")
        self.assertEqual(model_config["device_8bit"], 1)

    def test_bare_cuda_uses_current_device_for_quantized_load(self):
        model_config = {"low_resource": True}
        device = self.runtime_module.EmotionLLaMARuntime._configure_model_device(
            _FakeTorch(current_device=2, device_count=3), model_config, "cuda"
        )

        self.assertEqual(str(device), "cuda:2")
        self.assertEqual(model_config["device_8bit"], 2)

    def test_invalid_quantized_devices_fail_before_model_construction(self):
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            self.runtime_module.EmotionLLaMARuntime._configure_model_device(
                _FakeTorch(available=False, device_count=0),
                {"low_resource": True},
                "cpu",
            )
        with self.assertRaisesRegex(ValueError, "index 2 is unavailable"):
            self.runtime_module.EmotionLLaMARuntime._configure_model_device(
                _FakeTorch(device_count=2),
                {"low_resource": True},
                "cuda:2",
            )

    def test_quantized_llama_is_not_moved_after_construction(self):
        observed_llama_models = []
        llama_model = object()

        class FakeModel:
            def __init__(self):
                self.llama_model = llama_model

            def to(self, _device):
                observed_llama_models.append(self.llama_model)
                return self

        model = FakeModel()
        returned = self.runtime_module.EmotionLLaMARuntime._move_model_to_device(
            model, _FakeDevice("cuda:1"), low_resource=True
        )

        self.assertIs(returned, model)
        self.assertEqual(observed_llama_models, [None])
        self.assertIs(model.llama_model, llama_model)

        class FailingModel(FakeModel):
            def to(self, _device):
                self.assertion_state = self.llama_model
                raise RuntimeError("move failed")

        failing_model = FailingModel()
        with self.assertRaisesRegex(RuntimeError, "move failed"):
            self.runtime_module.EmotionLLaMARuntime._move_model_to_device(
                failing_model, _FakeDevice("cuda:1"), low_resource=True
            )
        self.assertIsNone(failing_model.assertion_state)
        self.assertIs(failing_model.llama_model, llama_model)

    def test_minigpt_v2_forwards_device_8bit_from_config(self):
        minigpt_v2 = _load_minigpt_v2_module()

        class CapturingMiniGPTv2(minigpt_v2.MiniGPTv2):
            def __init__(self, **kwargs):
                self.arguments = kwargs

        model = CapturingMiniGPTv2.from_config(
            {
                "image_size": 448,
                "llama_model": "llama",
                "low_resource": True,
                "device_8bit": 3,
            }
        )

        self.assertEqual(model.arguments["device_8bit"], 3)


class ChatAudioEncoderTests(unittest.TestCase):
    def test_audio_preload_reuses_extractor_and_model(self):
        conversation, transformers = _load_conversation_module()
        extractor_paths = []
        model_paths = []

        class FakeFeatureExtractor:
            @classmethod
            def from_pretrained(cls, path):
                extractor_paths.append(path)
                return cls()

        class FakeHubertInstance:
            def __init__(self):
                self.eval_calls = 0

            def eval(self):
                self.eval_calls += 1
                return self

        class FakeHubertModel:
            @classmethod
            def from_pretrained(cls, path):
                model_paths.append(path)
                return FakeHubertInstance()

        conversation.Wav2Vec2FeatureExtractor = FakeFeatureExtractor
        transformers.HubertModel = FakeHubertModel
        chat = conversation.Chat(
            model=object(),
            vis_processor=object(),
            stopping_criteria=(),
            audio_model_path="audio-model",
        )

        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            first = chat.load_audio_encoder()
            second = chat.load_audio_encoder()

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertEqual(extractor_paths, ["audio-model"])
        self.assertEqual(model_paths, ["audio-model"])
        self.assertEqual(first[1].eval_calls, 1)


class GradioClientImportTests(unittest.TestCase):
    def test_import_does_not_construct_runtime_or_launch_interface(self):
        runtime_calls = []
        interface_calls = []

        class FakeRuntime:
            def __init__(self, *_args, **_kwargs):
                runtime_calls.append(True)

        gradio = types.ModuleType("gradio")

        def forbidden_interface(*_args, **_kwargs):
            interface_calls.append(True)
            raise AssertionError("Gradio interface was built during import")

        gradio.Interface = forbidden_interface
        minigpt4 = types.ModuleType("minigpt4")
        minigpt4.__path__ = []
        inference = types.ModuleType("minigpt4.inference")
        inference.EmotionLLaMARuntime = FakeRuntime

        _load_source_module(
            Path("app_EmotionLlamaClient.py"),
            "gradio_client_under_test",
            {
                "gradio": gradio,
                "minigpt4": minigpt4,
                "minigpt4.inference": inference,
            },
        )

        self.assertEqual(runtime_calls, [])
        self.assertEqual(interface_calls, [])


if __name__ == "__main__":
    unittest.main()
