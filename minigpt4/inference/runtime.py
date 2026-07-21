"""Reusable video inference runtime with lazy model loading.

The Gradio applications historically owned model construction and conversation
setup.  Keeping that work in an importable runtime lets command-line tools and
other frontends load the models once and submit multiple independent videos.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence


DEFAULT_AUDIO_MODEL_PATH = "checkpoints/transformer/chinese-hubert-large"
DEFAULT_CFG_PATH = Path(__file__).resolve().parents[2] / "eval_configs" / "demo.yaml"


@dataclass(frozen=True)
class _RuntimeComponents:
    chat: Any
    conversation_factory: Callable[[], Any]
    device: str
    inference_context_factory: Callable[[], Any] = nullcontext


class EmotionLLaMARuntime:
    """Load Emotion-LLaMA lazily and reuse it for independent video requests.

    Args:
        cfg_path: Emotion-LLaMA demo configuration file.
        options: Optional OmegaConf dot-list overrides accepted by ``Config``.
        device: Explicit Torch device. When omitted, the runtime selects a
            device from Torch availability; the shipped checkpoints expect
            the CUDA environment documented by the project.
        dataset_name: Dataset config whose visual processor should be used.
        component_loader: Internal dependency-injection seam used by tests and
            lightweight integrations. The callable is invoked at most once.
    """

    def __init__(
        self,
        cfg_path=DEFAULT_CFG_PATH,
        *,
        options: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        dataset_name="feature_face_caption",
        component_loader: Optional[Callable[[], _RuntimeComponents]] = None,
    ):
        self.cfg_path = os.fspath(cfg_path)
        self.options = tuple(options or ())
        self.requested_device = device
        self.dataset_name = dataset_name
        self._component_loader = component_loader
        self._components = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def is_loaded(self):
        """Whether the model runtime has already been initialized."""

        return self._components is not None

    @property
    def device(self):
        """Resolved runtime device, or the requested value before loading."""

        if self._components is not None:
            return self._components.device
        return self.requested_device

    def load(self):
        """Initialize the model stack once and return this runtime."""

        if self._components is None:
            with self._load_lock:
                if self._components is None:
                    loader = self._component_loader or self._load_components
                    components = loader()
                    if not isinstance(components, _RuntimeComponents):
                        try:
                            components = _RuntimeComponents(*components)
                        except (TypeError, ValueError) as exc:
                            raise TypeError(
                                "component_loader must return chat, "
                                "conversation_factory, device, and optionally "
                                "an inference context factory"
                            ) from exc
                    self._components = components
        return self

    def analyze(
        self,
        video_path,
        prompt,
        *,
        temperature=0.2,
        max_new_tokens=500,
        max_length=2000,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.05,
        length_penalty=1.0,
        seed=None,
    ):
        """Analyze one video with a fresh conversation.

        The model, visual processor, and audio encoder are reused across calls.
        Requests are serialized because ``Chat`` and GPU generation are not
        thread-safe. This is sequential inference, not tensor batching.
        """

        resolved_video = Path(video_path).expanduser().resolve()
        if not resolved_video.is_file():
            raise FileNotFoundError(
                "Video file does not exist: {}".format(resolved_video)
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        self._validate_generation_options(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
        if seed is not None:
            self._validate_seed(seed)

        self.load()
        with self._inference_lock:
            components = self._components
            conversation = components.conversation_factory()
            image_list = []
            if seed is not None:
                self._set_seed(seed)
            with components.inference_context_factory():
                components.chat.upload_img(
                    str(resolved_video), conversation, image_list
                )
                components.chat.ask(prompt, conversation)
                components.chat.encode_img(image_list)
                answer = components.chat.answer(
                    conv=conversation,
                    img_list=image_list,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    num_beams=num_beams,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                )[0]
        return answer

    def load_audio_encoder(self):
        """Eagerly initialize the reusable HuBERT stack.

        Batch frontends can call this after ``load`` so configuration or model
        errors fail once before per-record processing begins.
        """

        self.load()
        with self._inference_lock:
            self._components.chat.load_audio_encoder()
        return self

    @staticmethod
    def _validate_seed(seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @classmethod
    def _set_seed(cls, seed):
        cls._validate_seed(seed)

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _validate_generation_options(**options):
        positive_integer_options = ("max_new_tokens", "max_length", "num_beams")
        for name in positive_integer_options:
            value = options[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))

        min_length = options["min_length"]
        if (
            isinstance(min_length, bool)
            or not isinstance(min_length, int)
            or min_length < 0
        ):
            raise ValueError("min_length must be a non-negative integer")
        if min_length > options["max_new_tokens"]:
            raise ValueError("min_length cannot exceed max_new_tokens")

        bounded_values = (
            (
                "temperature",
                options["temperature"],
                lambda value: value > 0,
                "greater than zero",
            ),
            (
                "top_p",
                options["top_p"],
                lambda value: 0 < value <= 1,
                "in the interval (0, 1]",
            ),
            (
                "repetition_penalty",
                options["repetition_penalty"],
                lambda value: value > 0,
                "greater than zero",
            ),
            (
                "length_penalty",
                options["length_penalty"],
                lambda value: value > 0,
                "greater than zero",
            ),
        )
        for name, value, predicate, requirement in bounded_values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not predicate(value)
            ):
                raise ValueError("{} must be {}".format(name, requirement))

    def _load_components(self):
        cfg_path = Path(self.cfg_path).expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(
                "Inference config file does not exist: {}".format(cfg_path)
            )

        import torch

        # Import registration modules only when the runtime is first used.
        import minigpt4.datasets.builders  # noqa: F401
        import minigpt4.models  # noqa: F401
        import minigpt4.processors  # noqa: F401
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import (
            CONV_VISION_minigptv2,
            Chat,
        )

        args = SimpleNamespace(cfg_path=str(cfg_path), options=list(self.options))
        cfg = Config(args)
        requested_device = self.requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_config = cfg.model_cfg
        repo_root = Path(registry.get_path("repo_root")).resolve()
        self._normalize_model_paths(model_config, repo_root)
        device = self._configure_model_device(
            torch, model_config, requested_device
        )
        model_cls = registry.get_model_class(model_config.arch)
        if model_cls is None:
            raise ValueError(
                "Unknown model architecture: {}".format(model_config.arch)
            )
        model = model_cls.from_config(model_config)
        self._move_model_to_device(
            model,
            device,
            low_resource=bool(model_config.get("low_resource", False)),
        )
        model.eval()

        if self.dataset_name not in cfg.datasets_cfg:
            raise ValueError(
                "Inference config does not define dataset {!r}".format(
                    self.dataset_name
                )
            )
        dataset_config = cfg.datasets_cfg.get(self.dataset_name)
        vis_processor_cfg = dataset_config.vis_processor.train
        processor_cls = registry.get_processor_class(vis_processor_cfg.name)
        if processor_cls is None:
            raise ValueError(
                "Unknown visual processor: {}".format(vis_processor_cfg.name)
            )
        vis_processor = processor_cls.from_config(vis_processor_cfg)

        audio_model_path = model_config.get(
            "audio_model_path", DEFAULT_AUDIO_MODEL_PATH
        )
        audio_model_path = Path(os.fspath(audio_model_path)).expanduser()
        if not audio_model_path.is_absolute():
            audio_model_path = repo_root / audio_model_path
        chat = Chat(
            model,
            vis_processor,
            device=device,
            audio_model_path=os.fspath(audio_model_path.resolve()),
        )
        return _RuntimeComponents(
            chat=chat,
            conversation_factory=CONV_VISION_minigptv2.copy,
            device=str(device),
            inference_context_factory=torch.inference_mode,
        )

    @staticmethod
    def _resolve_torch_device(torch, requested_device):
        """Return a validated, explicit Torch device.

        ``torch.device("cuda")`` deliberately leaves the index unspecified.
        Resolve it before model construction so an 8-bit ``device_map`` and
        the remaining model components always target the same logical GPU.
        """

        try:
            device = torch.device(requested_device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "Invalid Torch device {!r}: {}".format(requested_device, exc)
            ) from exc

        if device.type != "cuda":
            return device
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device {!r} was requested, but CUDA is not available".format(
                    requested_device
                )
            )

        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise ValueError(
                "CUDA device index {} is unavailable; {} device(s) are visible".format(
                    index, device_count
                )
            )
        return torch.device("cuda:{}".format(index))

    @classmethod
    def _configure_model_device(cls, torch, model_config, requested_device):
        device = cls._resolve_torch_device(torch, requested_device)
        if not bool(model_config.get("low_resource", False)):
            return device
        if device.type != "cuda":
            raise ValueError(
                "low_resource=true requires a CUDA device because the Llama "
                "model is loaded in 8-bit; select cuda[:N] or set "
                "model.low_resource=false"
            )

        # Quantized weights must be placed on their final device while they
        # are constructed; moving them between devices afterward is unsupported.
        model_config["device_8bit"] = device.index
        return device

    @staticmethod
    def _move_model_to_device(model, device, *, low_resource):
        if not low_resource:
            model.to(device)
            return model

        llama_model = getattr(model, "llama_model", None)
        if llama_model is None:
            raise ValueError(
                "low_resource model did not expose its quantized llama_model"
            )

        # ``nn.Module.to`` recursively applies to child modules and bypasses
        # the guard implemented by quantized Transformers models. Temporarily
        # detach the already-placed 8-bit Llama so only the visual/projector
        # components move to the selected device.
        model.llama_model = None
        try:
            model.to(device)
        finally:
            model.llama_model = llama_model
        return model

    @staticmethod
    def _normalize_model_paths(model_config, repo_root):
        """Resolve documented local model paths independently of the CWD."""

        ckpt_path = model_config.get("ckpt")
        if ckpt_path:
            path = Path(os.fspath(ckpt_path)).expanduser()
            if not path.is_absolute():
                path = repo_root / path
            model_config["ckpt"] = os.fspath(path.resolve())

        llama_model = model_config.get("llama_model")
        if not llama_model:
            return
        path = Path(os.fspath(llama_model)).expanduser()
        if path.is_absolute():
            model_config["llama_model"] = os.fspath(path.resolve())
            return

        normalized = os.fspath(llama_model).replace("\\", "/")
        local_candidate = repo_root / path
        if (
            normalized.startswith(("./", "../", "checkpoints/"))
            or local_candidate.exists()
        ):
            model_config["llama_model"] = os.fspath(local_candidate.resolve())
