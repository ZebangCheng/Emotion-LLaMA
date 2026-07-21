"""Sequential, resumable batch inference for Emotion-LLaMA."""

import argparse
import ast
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


SCHEMA_VERSION = 1
RUNTIME_FINGERPRINT_VERSION = 2
DEFAULT_EXTENSIONS = ".mp4,.avi,.mov,.mkv"
DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "eval_configs" / "demo.yaml"
DEFAULT_AUDIO_MODEL_PATH = "checkpoints/transformer/chinese-hubert-large"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
HASH_CHUNK_SIZE = 4 * 1024 * 1024
MODEL_ARTIFACT_KEYS = frozenset(
    (
        "audio_model_path",
        "ckpt",
        "finetuned",
        "llama_model",
        "pretrained",
        "q_former_model",
    )
)


class BatchInferenceError(Exception):
    """Base class for expected command-line failures."""

    exit_code = 2


class InputContractError(BatchInferenceError):
    """The CLI arguments or input records are invalid."""


class RuntimeInitializationError(BatchInferenceError):
    """The shared model runtime could not be initialized."""

    exit_code = 3


class OutputStateError(BatchInferenceError):
    """The output JSONL is corrupt or cannot be updated safely."""

    exit_code = 4


class _DuplicateJSONKeyError(ValueError):
    pass


class _VideoStateChangedError(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchItem:
    item_id: str
    video: str
    resolved_video: Path
    prompt: str
    source: dict
    request_hash: str = ""
    video_fingerprint: str = None

    def with_request_hash(self):
        payload = {
            "id": self.item_id,
            "resolved_video": str(self.resolved_video),
            "prompt": self.prompt,
        }
        return BatchItem(
            item_id=self.item_id,
            video=self.video,
            resolved_video=self.resolved_video,
            prompt=self.prompt,
            source=self.source,
            request_hash=_sha256_json(payload),
            video_fingerprint=_video_fingerprint(self.resolved_video),
        )


@dataclass(frozen=True)
class ExistingOutput:
    records: tuple
    repaired_tail: bool = False
    missing_final_newline: bool = False


def _video_fingerprint(path):
    try:
        if not path.is_file():
            return None
        return "sha256:" + _sha256_file(path)
    except OSError:
        return None


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError("duplicate key {!r}".format(key))
        result[key] = value
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run sequential Emotion-LLaMA inference while loading the model once."
        )
    )
    parser.add_argument(
        "--cfg-path",
        default=str(DEFAULT_CFG_PATH),
        help="path to the inference configuration file",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", help="infer one video")
    inputs.add_argument(
        "--input",
        "--manifest",
        dest="manifest",
        help="CSV or JSONL input manifest",
    )
    inputs.add_argument("--video-dir", help="infer videos from a directory")
    parser.add_argument("--output", required=True, help="output JSONL path")

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="fallback prompt for input records")
    prompt_group.add_argument("--prompt-file", help="UTF-8 file containing a prompt")
    parser.add_argument(
        "--force-prompt",
        action="store_true",
        help="override every manifest prompt with the global prompt",
    )

    parser.add_argument("--id", help="record id for --video")
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "jsonl"),
        default="auto",
        help="manifest format; auto uses the file extension",
    )
    parser.add_argument(
        "--base-dir",
        help="base directory for relative manifest video paths",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="scan --video-dir recursively",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXTENSIONS,
        help="comma-separated extensions used by --video-dir",
    )

    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="skip matching successful records and retry failed records",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output after input and runtime validation",
    )

    parser.add_argument("--device", help="Torch device, for example cuda or cuda:0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--options",
        nargs="+",
        help=(
            "concrete configuration overrides in xxx=yyy format; interpolation "
            "and model.arch/model.model_type overrides are not supported"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after writing the first per-record error",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-record tracebacks to stderr",
    )
    return parser


def _read_global_prompt(args):
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        try:
            prompt = prompt_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise InputContractError(
                "Cannot read prompt file {}: {}".format(prompt_path, exc)
            ) from exc
    else:
        prompt = None

    if prompt is not None:
        prompt = prompt.strip()
        if not prompt:
            raise InputContractError("The global prompt must not be empty")
    return prompt


def _validate_mode_arguments(args, global_prompt):
    manifest_only = {
        "--format": args.format != "auto",
        "--base-dir": args.base_dir is not None,
        "--force-prompt": args.force_prompt,
    }
    if args.manifest is None:
        invalid = [name for name, enabled in manifest_only.items() if enabled]
        if invalid:
            raise InputContractError(
                "{} can only be used with --manifest".format(", ".join(invalid))
            )

    if args.video is None and args.id is not None:
        raise InputContractError("--id can only be used with --video")
    if args.video_dir is None and args.recursive:
        raise InputContractError("--recursive can only be used with --video-dir")
    if args.force_prompt and global_prompt is None:
        raise InputContractError("--force-prompt requires --prompt or --prompt-file")
    if (args.video is not None or args.video_dir is not None) and global_prompt is None:
        raise InputContractError("--video and --video-dir require a global prompt")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise InputContractError("--seed must be non-negative")
    _validate_generation_arguments(args)


def _validate_generation_arguments(args):
    for name in ("max_new_tokens", "max_length", "num_beams"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InputContractError(
                "--{} must be a positive integer".format(name.replace("_", "-"))
            )

    if (
        isinstance(args.min_length, bool)
        or not isinstance(args.min_length, int)
        or args.min_length < 0
    ):
        raise InputContractError("--min-length must be a non-negative integer")
    if args.min_length > args.max_new_tokens:
        raise InputContractError("--min-length cannot exceed --max-new-tokens")

    bounded_values = (
        ("temperature", args.temperature, lambda value: value > 0, "greater than zero"),
        ("top-p", args.top_p, lambda value: 0 < value <= 1, "in the interval (0, 1]"),
        (
            "repetition-penalty",
            args.repetition_penalty,
            lambda value: value > 0,
            "greater than zero",
        ),
        (
            "length-penalty",
            args.length_penalty,
            lambda value: value > 0,
            "greater than zero",
        ),
    )
    for name, value, predicate, requirement in bounded_values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InputContractError("--{} must be numeric".format(name))
        if not math.isfinite(value) or not predicate(value):
            raise InputContractError("--{} must be {}".format(name, requirement))


def _resolve_path(value, base_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _default_manifest_id(video):
    return video.replace("\\", "/")


def _normalize_manifest_record(raw_record, source, base_dir, global_prompt, force):
    if not isinstance(raw_record, dict):
        raise InputContractError(
            "{} must contain a JSON object or CSV row".format(_source_label(source))
        )

    video = raw_record.get("video")
    if not isinstance(video, str) or not video.strip():
        raise InputContractError(
            "{} has an empty or non-string 'video' field".format(
                _source_label(source)
            )
        )
    video = video.strip()

    item_id = raw_record.get("id")
    if item_id is None or (isinstance(item_id, str) and not item_id.strip()):
        item_id = _default_manifest_id(video)
    if not isinstance(item_id, str) or not item_id.strip():
        raise InputContractError(
            "{} has an empty or non-string 'id' field".format(_source_label(source))
        )
    item_id = item_id.strip()

    row_prompt = raw_record.get("prompt")
    if force:
        prompt = global_prompt
    elif isinstance(row_prompt, str) and row_prompt.strip():
        prompt = row_prompt.strip()
    elif row_prompt is not None and not isinstance(row_prompt, str):
        raise InputContractError(
            "{} has a non-string 'prompt' field".format(_source_label(source))
        )
    else:
        prompt = global_prompt
    if prompt is None:
        raise InputContractError(
            "{} has no prompt and no global fallback was provided".format(
                _source_label(source)
            )
        )

    return BatchItem(
        item_id=item_id,
        video=video,
        resolved_video=_resolve_path(video, base_dir),
        prompt=prompt,
        source=source,
    ).with_request_hash()


def _source_label(source):
    return "{} record {}".format(source["input"], source["record"])


def _detect_manifest_format(path, requested_format):
    if requested_format != "auto":
        return requested_format
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return "csv"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    raise InputContractError(
        "Cannot infer manifest format from {!r}; use --format".format(path.name)
    )


def _load_csv_manifest(path, base_dir, global_prompt, force):
    try:
        items = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None or "video" not in reader.fieldnames:
                raise InputContractError("CSV manifest must contain a 'video' header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise InputContractError("CSV manifest contains duplicate headers")
            for row_number, row in enumerate(reader, start=1):
                values = [value for key, value in row.items() if key is not None]
                overflow_values = row.get(None) or []
                if all(
                    value is None or not value.strip()
                    for value in values + list(overflow_values)
                ):
                    continue
                if None in row:
                    raise InputContractError(
                        "CSV record {} contains more fields than the header".format(
                            row_number
                        )
                    )
                source = {
                    "mode": "csv",
                    "input": str(path),
                    "record": row_number,
                }
                items.append(
                    _normalize_manifest_record(
                        row, source, base_dir, global_prompt, force
                    )
                )
    except InputContractError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputContractError("Cannot read manifest {}: {}".format(path, exc)) from exc
    return items


def _load_jsonl_manifest(path, base_dir, global_prompt, force):
    try:
        items = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(
                        line, object_pairs_hook=_reject_duplicate_json_keys
                    )
                except (json.JSONDecodeError, _DuplicateJSONKeyError) as exc:
                    raise InputContractError(
                        "Invalid JSON in {} line {}: {}".format(
                            path, line_number, exc
                        )
                    ) from exc
                source = {
                    "mode": "jsonl",
                    "input": str(path),
                    "record": line_number,
                }
                items.append(
                    _normalize_manifest_record(
                        record, source, base_dir, global_prompt, force
                    )
                )
    except InputContractError:
        raise
    except (OSError, UnicodeError) as exc:
        raise InputContractError("Cannot read manifest {}: {}".format(path, exc)) from exc
    return items


def _parse_extensions(raw_extensions):
    extensions = []
    for value in raw_extensions.split(","):
        value = value.strip().casefold()
        if not value:
            continue
        if not value.startswith("."):
            value = "." + value
        extensions.append(value)
    if not extensions:
        raise InputContractError("--extensions must contain at least one extension")
    return frozenset(extensions)


def _load_directory_items(args, global_prompt):
    directory = Path(args.video_dir).expanduser().resolve()
    if not directory.is_dir():
        raise InputContractError("Video directory does not exist: {}".format(directory))
    extensions = _parse_extensions(args.extensions)
    candidates = directory.rglob("*") if args.recursive else directory.iterdir()
    videos = [
        path
        for path in candidates
        if path.is_file() and path.suffix.casefold() in extensions
    ]
    videos.sort(
        key=lambda path: (
            path.relative_to(directory).as_posix().casefold(),
            path.relative_to(directory).as_posix(),
        )
    )
    if not videos:
        raise InputContractError(
            "No matching videos found in directory: {}".format(directory)
        )

    items = []
    for record_number, video_path in enumerate(videos, start=1):
        relative_video = video_path.relative_to(directory).as_posix()
        items.append(
            BatchItem(
                item_id=relative_video,
                video=relative_video,
                resolved_video=video_path.resolve(),
                prompt=global_prompt,
                source={
                    "mode": "directory",
                    "input": str(directory),
                    "record": record_number,
                },
            ).with_request_hash()
        )
    return items


def load_items(args):
    global_prompt = _read_global_prompt(args)
    _validate_mode_arguments(args, global_prompt)

    if args.video is not None:
        video_path = Path(args.video).expanduser().resolve()
        item_id = args.id.strip() if args.id is not None else video_path.name
        if not item_id:
            raise InputContractError("--id must not be empty")
        items = [
            BatchItem(
                item_id=item_id,
                video=args.video,
                resolved_video=video_path,
                prompt=global_prompt,
                source={
                    "mode": "video",
                    "input": str(video_path),
                    "record": 1,
                },
            ).with_request_hash()
        ]
    elif args.video_dir is not None:
        items = _load_directory_items(args, global_prompt)
    else:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise InputContractError(
                "Manifest file does not exist: {}".format(manifest_path)
            )
        base_dir = (
            Path(args.base_dir).expanduser().resolve()
            if args.base_dir is not None
            else manifest_path.parent
        )
        manifest_format = _detect_manifest_format(manifest_path, args.format)
        if manifest_format == "csv":
            items = _load_csv_manifest(
                manifest_path, base_dir, global_prompt, args.force_prompt
            )
        else:
            items = _load_jsonl_manifest(
                manifest_path, base_dir, global_prompt, args.force_prompt
            )

    if not items:
        raise InputContractError("The selected input contains no records")
    _validate_unique_ids(items)
    return items


def _validate_unique_ids(items):
    seen = {}
    for item in items:
        if item.item_id in seen:
            raise InputContractError(
                "Duplicate id {!r}: {} and {}".format(
                    item.item_id,
                    _source_label(seen[item.item_id].source),
                    _source_label(item.source),
                )
            )
        seen[item.item_id] = item


def _sha256_json(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    observed = path.stat()
    before_signature = _file_stat_signature(before)
    if (
        _file_stat_signature(after) != before_signature
        or _file_stat_signature(observed) != before_signature
    ):
        raise OSError(
            "File changed while it was being fingerprinted: {}".format(path)
        )
    return digest.hexdigest()


def _file_stat_signature(stat_result):
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _read_yaml_mapping(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeInitializationError(
            "PyYAML is required to fingerprint model artifacts"
        ) from exc

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeInitializationError(
            "Cannot read model configuration {}: {}".format(path, exc)
        ) from exc
    return value if isinstance(value, dict) else {}


def _model_option_overrides(options):
    options = list(options or ())
    if not options:
        return {}

    if "=" in options[0]:
        pairs = []
        for option in options:
            key, separator, value = option.partition("=")
            if separator:
                pairs.append((key, value))
    else:
        pairs = list(zip(options[0::2], options[1::2]))

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeInitializationError(
            "PyYAML is required to fingerprint model artifacts"
        ) from exc

    overrides = {}
    for key, value in pairs:
        if not key.startswith("model."):
            continue
        model_key = key[len("model.") :]
        if not model_key or "." in model_key:
            continue
        if model_key in ("arch", "model_type"):
            raise RuntimeInitializationError(
                "Overriding model.{} through --options is not supported for safe "
                "fingerprinting; select it in the YAML configuration".format(
                    model_key
                )
            )
        try:
            overrides[model_key] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise RuntimeInitializationError(
                "Cannot parse configuration override {!r}: {}".format(key, exc)
            ) from exc
    return overrides


def _default_model_config_path(repo_root, arch, model_type):
    if not isinstance(arch, str) or not isinstance(model_type, str):
        return None

    models_root = repo_root / "minigpt4" / "models"
    try:
        source_paths = sorted(models_root.glob("*.py"))
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                registered_names = []
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == "register_model"
                        and decorator.args
                        and isinstance(decorator.args[0], ast.Constant)
                    ):
                        registered_names.append(decorator.args[0].value)
                if arch not in registered_names:
                    continue

                for statement in node.body:
                    if not isinstance(statement, ast.Assign):
                        continue
                    if not any(
                        isinstance(target, ast.Name)
                        and target.id == "PRETRAINED_MODEL_CONFIG_DICT"
                        for target in statement.targets
                    ):
                        continue
                    mapping = ast.literal_eval(statement.value)
                    relative_path = mapping.get(model_type)
                    if relative_path:
                        return (repo_root / "minigpt4" / relative_path).resolve()
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        raise RuntimeInitializationError(
            "Cannot resolve the default model configuration: {}".format(exc)
        ) from exc
    return None


def _contains_interpolation(value):
    if isinstance(value, str):
        return "${" in value
    if isinstance(value, dict):
        return any(
            _contains_interpolation(key) or _contains_interpolation(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_interpolation(item) for item in value)
    return False


def _effective_model_settings(args, cfg_path, repo_root):
    user_configuration = _read_yaml_mapping(cfg_path)
    if _contains_interpolation(user_configuration) or _contains_interpolation(
        list(args.options or ())
    ):
        raise RuntimeInitializationError(
            "Interpolated inference configuration is not supported for safe "
            "fingerprinting; provide concrete configuration values"
        )
    user_model = user_configuration.get("model")
    if user_model is None:
        user_model = {}
    elif not isinstance(user_model, dict):
        raise RuntimeInitializationError(
            "The model configuration must be a mapping for safe fingerprinting"
        )
    overrides = _model_option_overrides(args.options)
    arch = overrides.get("arch", user_model.get("arch"))
    model_type = overrides.get("model_type", user_model.get("model_type"))
    default_path = _default_model_config_path(
        repo_root, arch=arch, model_type=model_type
    )
    if arch is not None and model_type is not None and default_path is None:
        raise RuntimeInitializationError(
            "Cannot fingerprint default model assets for architecture {!r} and "
            "model type {!r}".format(arch, model_type)
        )
    settings = {}
    if default_path is not None:
        default_configuration = _read_yaml_mapping(default_path)
        default_model = default_configuration.get("model")
        if isinstance(default_model, dict):
            if _contains_interpolation(default_model):
                raise RuntimeInitializationError(
                    "Interpolated default model configuration is not supported "
                    "for safe fingerprinting"
                )
            settings.update(default_model)
    settings.update(user_model)
    settings.update(overrides)
    settings.setdefault("audio_model_path", DEFAULT_AUDIO_MODEL_PATH)
    return settings


def _resolve_model_artifact_path(key, value, repo_root):
    if not isinstance(value, str) or not value.strip():
        return None
    if "${" in value:
        raise RuntimeInitializationError(
            "Interpolated model artifact path {!r} is not supported; provide a "
            "concrete model.{} value".format(value, key)
        )
    normalized = value.replace("\\", "/")
    if normalized.casefold().startswith(("http://", "https://")):
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = repo_root / path
    is_explicit_local = normalized.startswith(("./", "../", "checkpoints/"))
    if key in ("audio_model_path", "ckpt") or is_explicit_local or candidate.exists():
        return candidate.resolve()
    return None


def _path_content_identity(path):
    if not path.exists():
        return {"kind": "missing"}
    try:
        if path.is_file():
            return {
                "kind": "file",
                "sha256": _sha256_file(path),
            }
        if not path.is_dir():
            return {"kind": "unsupported"}

        digest = hashlib.sha256(b"model-artifact-tree-v1\0")
        file_count = 0
        entries = sorted(
            path.rglob("*"),
            key=lambda entry: entry.relative_to(path).as_posix(),
        )
        for entry in entries:
            relative_path = entry.relative_to(path).as_posix()
            if entry.is_symlink():
                digest.update(b"symlink\0")
                digest.update(relative_path.encode("utf-8"))
                digest.update(b"\0")
                digest.update(os.fspath(os.readlink(entry)).encode("utf-8"))
                digest.update(b"\0")
                if entry.is_dir():
                    raise RuntimeInitializationError(
                        "Directory symlinks are not supported in model artifact {}: "
                        "{}".format(path, relative_path)
                    )
            if entry.is_dir():
                continue
            if not entry.is_file():
                raise RuntimeInitializationError(
                    "Unsupported entry in model artifact {}: {}".format(
                        path, relative_path
                    )
                )
            digest.update(b"file\0")
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_file(entry).encode("ascii"))
            digest.update(b"\0")
            file_count += 1
        return {
            "kind": "directory",
            "file_count": file_count,
            "sha256": digest.hexdigest(),
        }
    except OSError as exc:
        raise RuntimeInitializationError(
            "Cannot fingerprint model artifact {}: {}".format(path, exc)
        ) from exc


def _model_artifact_snapshots(args, cfg_path):
    repo_root = Path(__file__).resolve().parent
    settings = _effective_model_settings(args, cfg_path, repo_root)
    identities = {}
    snapshots = []
    for key in sorted(MODEL_ARTIFACT_KEYS):
        value = settings.get(key)
        path = _resolve_model_artifact_path(key, value, repo_root)
        if path is None:
            if isinstance(value, str) and value.strip():
                snapshots.append(
                    {
                        "key": key,
                        "reference": value,
                        "identity": {"kind": "external-reference"},
                    }
                )
            continue
        path_label = str(path)
        if path_label not in identities:
            identities[path_label] = _path_content_identity(path)
        snapshots.append(
            {
                "key": key,
                "reference": value,
                "path": path_label,
                "identity": identities[path_label],
            }
        )
    return snapshots


def _generation_options(args):
    return {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length,
        "num_beams": args.num_beams,
        "min_length": args.min_length,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
    }


def _device_fingerprint(requested_device, torch_module=None):
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return {
                "requested": requested_device,
                "resolved": "torch-unavailable",
            }

    selected_device = requested_device or (
        "cuda" if torch_module.cuda.is_available() else "cpu"
    )
    try:
        device = torch_module.device(selected_device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeInitializationError(
            "Invalid Torch device {!r}: {}".format(selected_device, exc)
        ) from exc

    result = {
        "requested": requested_device,
        "resolved": str(device),
    }
    if device.type != "cuda":
        return result

    result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not torch_module.cuda.is_available():
        result["available"] = False
        return result
    index = device.index
    if index is None:
        index = torch_module.cuda.current_device()
    result["resolved"] = "cuda:{}".format(index)
    result["available"] = index < torch_module.cuda.device_count()
    return result


def _run_fingerprint(args):
    cfg_path = Path(args.cfg_path).expanduser().resolve()
    try:
        config_bytes = cfg_path.read_bytes()
    except OSError as exc:
        raise RuntimeInitializationError(
            "Cannot read inference config {}: {}".format(cfg_path, exc)
        ) from exc
    payload = {
        "fingerprint_version": RUNTIME_FINGERPRINT_VERSION,
        "implementation_sha256": _implementation_fingerprint(),
        "config_sha256": hashlib.sha256(
            config_bytes.replace(b"\r\n", b"\n")
        ).hexdigest(),
        "model_artifacts": _model_artifact_snapshots(args, cfg_path),
        "options": list(args.options or ()),
        "device": _device_fingerprint(args.device),
        "generation": _generation_options(args),
        "seed": args.seed,
    }
    return _sha256_json(payload)


def _implementation_fingerprint():
    repo_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    try:
        relative_paths = [Path("batch_infer.py")]
        relative_paths.extend(
            path.relative_to(repo_root)
            for path in (repo_root / "minigpt4").rglob("*")
            if path.is_file()
            and path.suffix.casefold() in (".py", ".yaml", ".yml")
        )
        relative_paths.sort(key=lambda path: path.as_posix())
        for relative_path in relative_paths:
            digest.update(relative_path.as_posix().encode("utf-8"))
            digest.update(b"\0")
            source_bytes = (repo_root / relative_path).read_bytes()
            if (
                relative_path.suffix.casefold() in (".yaml", ".yml")
                and b"${" in source_bytes
            ):
                raise RuntimeInitializationError(
                    "Interpolated default configuration is not supported for safe "
                    "fingerprinting: {}".format(relative_path)
                )
            digest.update(source_bytes.replace(b"\r\n", b"\n"))
            digest.update(b"\0")
    except OSError as exc:
        raise RuntimeInitializationError(
            "Cannot fingerprint inference implementation: {}".format(exc)
        ) from exc
    return digest.hexdigest()


def _validate_output_record(record, path, line_number):
    label = "{} line {}".format(path, line_number)
    if not isinstance(record, dict):
        raise OutputStateError("{} is not a JSON object".format(label))
    required = {
        "schema_version",
        "run_fingerprint",
        "request_hash",
        "video_fingerprint",
        "id",
        "video",
        "resolved_video",
        "prompt",
        "source",
        "status",
        "answer",
        "error",
        "duration_ms",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise OutputStateError(
            "{} is missing field(s): {}".format(label, ", ".join(missing))
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise OutputStateError(
            "{} uses unsupported schema_version {!r}".format(
                label, record["schema_version"]
            )
        )
    if not isinstance(record["id"], str) or not record["id"]:
        raise OutputStateError("{} has an invalid id".format(label))
    for name in (
        "run_fingerprint",
        "request_hash",
        "video",
        "resolved_video",
        "prompt",
    ):
        if not isinstance(record[name], str) or not record[name]:
            raise OutputStateError("{} has an invalid {}".format(label, name))
    if not isinstance(record["source"], dict):
        raise OutputStateError("{} has an invalid source".format(label))
    video_fingerprint = record["video_fingerprint"]
    if video_fingerprint is not None and (
        not isinstance(video_fingerprint, str) or not video_fingerprint
    ):
        raise OutputStateError("{} has an invalid video_fingerprint".format(label))
    if (
        isinstance(record["duration_ms"], bool)
        or not isinstance(record["duration_ms"], int)
        or record["duration_ms"] < 0
    ):
        raise OutputStateError("{} has an invalid duration_ms".format(label))
    if record["status"] not in (STATUS_SUCCESS, STATUS_ERROR):
        raise OutputStateError("{} has an invalid status".format(label))
    if record["status"] == STATUS_SUCCESS:
        if (
            not isinstance(record["answer"], str)
            or record["error"] is not None
            or video_fingerprint is None
        ):
            raise OutputStateError("{} has an invalid success payload".format(label))
    else:
        error = record["error"]
        if record["answer"] is not None or not isinstance(error, dict):
            raise OutputStateError("{} has an invalid error payload".format(label))
        for name in ("stage", "type", "message"):
            if not isinstance(error.get(name), str) or not error[name]:
                raise OutputStateError(
                    "{} has an invalid error.{}".format(label, name)
                )
    return record


def read_existing_output(path, *, repair_tail):
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise OutputStateError("Cannot read output {}: {}".format(path, exc)) from exc
    if not data:
        return ExistingOutput(())

    records = []
    seen_ids = set()
    segments = data.splitlines(keepends=True)
    repaired = False
    missing_final_newline = False
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        has_newline = segment.endswith((b"\n", b"\r"))
        raw_line = segment.rstrip(b"\r\n")
        if not raw_line:
            raise OutputStateError(
                "{} line {} is empty".format(path, index + 1)
            )
        try:
            decoded = raw_line.decode("utf-8")
            record = json.loads(
                decoded, object_pairs_hook=_reject_duplicate_json_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if repair_tail and is_last and not has_newline:
                repaired = True
                break
            raise OutputStateError(
                "Invalid JSON in {} line {}: {}".format(path, index + 1, exc)
            ) from exc
        except _DuplicateJSONKeyError as exc:
            raise OutputStateError(
                "Invalid JSON in {} line {}: {}".format(path, index + 1, exc)
            ) from exc

        record = _validate_output_record(record, path, index + 1)

        if record["id"] in seen_ids:
            raise OutputStateError(
                "Output {} contains duplicate id {!r}".format(path, record["id"])
            )
        seen_ids.add(record["id"])
        records.append(record)
        if is_last and not has_newline:
            missing_final_newline = True

    return ExistingOutput(
        tuple(records),
        repaired_tail=repaired,
        missing_final_newline=missing_final_newline,
    )


def _atomic_rewrite(path, records):
    descriptor = None
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            for record in records:
                handle.write(_json_line(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
        temporary_path = None
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise OutputStateError("Cannot update output {}: {}".format(path, exc)) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _write_record(handle, record):
    try:
        handle.write(_json_line(record))
        handle.flush()
        os.fsync(handle.fileno())
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise OutputStateError("Cannot write output record: {}".format(exc)) from exc


def _json_line(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def _append_record(path, record):
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            _write_record(handle, record)
    except OutputStateError:
        raise
    except OSError as exc:
        raise OutputStateError("Cannot open output {}: {}".format(path, exc)) from exc


def _persist_record(path, records, record_positions, record):
    position = record_positions.get(record["id"])
    if position is None:
        _append_record(path, record)
        record_positions[record["id"]] = len(records)
        records.append(record)
        return

    updated_records = list(records)
    updated_records[position] = record
    _atomic_rewrite(path, updated_records)
    records[:] = updated_records


def _output_lock_path(output_path):
    return output_path.with_name("." + output_path.name + ".lock")


def _lock_file(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _output_lock(output_path):
    lock_path = _output_lock_path(output_path)
    handle = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        _lock_file(handle)
    except OSError as exc:
        if handle is not None:
            handle.close()
        raise OutputStateError(
            "Cannot acquire output lock {}: {}".format(lock_path, exc)
        ) from exc

    try:
        yield
    finally:
        try:
            _unlock_file(handle)
        except OSError:
            pass
        handle.close()


def _prepare_existing_output(args, output_path, items, run_fingerprint):
    if not output_path.exists():
        return (), items, 0, False, False
    if not output_path.is_file():
        raise OutputStateError("Output path is not a file: {}".format(output_path))
    if args.overwrite:
        return (), items, 0, False, False
    if not args.resume:
        raise InputContractError(
            "Output already exists; use --resume or --overwrite: {}".format(
                output_path
            )
        )

    existing = read_existing_output(output_path, repair_tail=True)
    items_by_id = {item.item_id: item for item in items}
    for previous in existing.records:
        item = items_by_id.get(previous["id"])
        if item is None:
            raise InputContractError(
                "Cannot resume: output contains id {!r} not present in the input".format(
                    previous["id"]
                )
            )
        if previous["run_fingerprint"] != run_fingerprint:
            raise InputContractError(
                "Cannot resume id {!r} with different runtime settings".format(
                    previous["id"]
                )
            )
        if previous["request_hash"] != item.request_hash:
            raise InputContractError(
                "Cannot resume id {!r}: video or prompt changed".format(
                    previous["id"]
                )
            )
        if (
            previous["status"] == STATUS_SUCCESS
            and previous["video_fingerprint"] != item.video_fingerprint
        ):
            raise InputContractError(
                "Cannot resume id {!r}: video contents changed".format(
                    previous["id"]
                )
            )

    existing_by_id = {record["id"]: record for record in existing.records}
    pending = []
    skipped = 0
    for item in items:
        previous = existing_by_id.get(item.item_id)
        if previous is None:
            pending.append(item)
            continue
        if previous["status"] == STATUS_SUCCESS:
            skipped += 1
        else:
            pending.append(item)

    needs_rewrite = existing.repaired_tail or existing.missing_final_newline
    return (
        existing.records,
        pending,
        skipped,
        existing.repaired_tail,
        needs_rewrite,
    )


def _item_seed(base_seed, request_hash):
    digest = hashlib.sha256(
        "{}:{}".format(base_seed, request_hash).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _error_payload(stage, exc):
    message = str(exc).strip() or exc.__class__.__name__
    return {
        "stage": stage,
        "type": exc.__class__.__name__,
        "message": message,
    }


def _result_record(item, run_fingerprint, status, duration_ms, answer=None, error=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "request_hash": item.request_hash,
        "video_fingerprint": item.video_fingerprint,
        "id": item.item_id,
        "video": item.video,
        "resolved_video": str(item.resolved_video),
        "prompt": item.prompt,
        "source": item.source,
        "status": status,
        "answer": answer,
        "error": error,
        "duration_ms": duration_ms,
    }


def _is_out_of_memory(exc):
    return exc.__class__.__name__ == "OutOfMemoryError" or (
        "out of memory" in str(exc).casefold()
    )


def _create_runtime(args):
    # Keep --help and input validation independent of the optional ML stack.
    from minigpt4.inference import EmotionLLaMARuntime

    return EmotionLLaMARuntime(
        args.cfg_path,
        options=args.options,
        device=args.device,
    )


def run(args, *, runtime_factory=None):
    items = load_items(args)
    output_path = Path(args.output).expanduser().resolve()
    protected_inputs = {item.resolved_video for item in items}
    protected_inputs.add(Path(args.cfg_path).expanduser().resolve())
    if args.manifest is not None:
        protected_inputs.add(Path(args.manifest).expanduser().resolve())
    if args.prompt_file is not None:
        protected_inputs.add(Path(args.prompt_file).expanduser().resolve())
    if output_path in protected_inputs:
        raise InputContractError(
            "Output must not overwrite an input file: {}".format(output_path)
        )
    lock_path = _output_lock_path(output_path)
    if lock_path in protected_inputs:
        raise InputContractError(
            "Output lock must not overwrite an input file: {}".format(lock_path)
        )

    with _output_lock(output_path):
        return _run_locked(args, items, output_path, runtime_factory)


def _run_locked(args, items, output_path, runtime_factory):
    run_fingerprint = _run_fingerprint(args)
    (
        existing_records,
        pending,
        skipped,
        repaired_tail,
        needs_rewrite,
    ) = _prepare_existing_output(args, output_path, items, run_fingerprint)

    runnable_items = [
        item for item in pending if item.video_fingerprint is not None
    ]
    runtime = None
    if runnable_items:
        factory = runtime_factory or (lambda: _create_runtime(args))
        try:
            runtime = factory()
            runtime.load()
            runtime.load_audio_encoder()
            if _run_fingerprint(args) != run_fingerprint:
                raise RuntimeError(
                    "Inference implementation, device, configuration, or model "
                    "assets changed while the runtime was loading"
                )
        except Exception as exc:
            raise RuntimeInitializationError(
                "Cannot initialize inference runtime: {}".format(exc)
            ) from exc

    if args.overwrite:
        _atomic_rewrite(output_path, ())
        records = []
    elif args.resume and output_path.exists() and needs_rewrite:
        _atomic_rewrite(output_path, existing_records)
        records = list(existing_records)
    elif not output_path.exists():
        _atomic_rewrite(output_path, ())
        records = []
    else:
        records = list(existing_records)

    record_positions = {
        record["id"]: index for index, record in enumerate(records)
    }
    succeeded = 0
    failed = 0
    global_failure = False
    for item in pending:
        started = time.perf_counter()
        try:
            if not item.resolved_video.is_file():
                raise FileNotFoundError(
                    "Video file does not exist: {}".format(item.resolved_video)
                )
            if item.video_fingerprint is None:
                raise OSError(
                    "Video file cannot be read for content fingerprinting: {}".format(
                        item.resolved_video
                    )
                )
            if _video_fingerprint(item.resolved_video) != item.video_fingerprint:
                raise _VideoStateChangedError(
                    "Video changed after the input manifest was loaded: {}".format(
                        item.resolved_video
                    )
                )

            answer = runtime.analyze(
                item.resolved_video,
                item.prompt,
                seed=_item_seed(args.seed, item.request_hash),
                **_generation_options(args),
            )
            if not isinstance(answer, str):
                raise TypeError("Inference runtime returned a non-string answer")
            if _video_fingerprint(item.resolved_video) != item.video_fingerprint:
                raise _VideoStateChangedError(
                    "Video changed while inference was running: {}".format(
                        item.resolved_video
                    )
                )
            record = _result_record(
                item,
                run_fingerprint,
                STATUS_SUCCESS,
                duration_ms=round((time.perf_counter() - started) * 1000),
                answer=answer,
            )
            succeeded += 1
        except Exception as exc:
            if args.verbose:
                traceback.print_exc(file=sys.stderr)
            stage = (
                "validate"
                if isinstance(exc, (OSError, _VideoStateChangedError))
                else "inference"
            )
            record = _result_record(
                item,
                run_fingerprint,
                STATUS_ERROR,
                duration_ms=round((time.perf_counter() - started) * 1000),
                error=_error_payload(stage, exc),
            )
            failed += 1
            if _is_out_of_memory(exc):
                global_failure = True
        _persist_record(output_path, records, record_positions, record)
        if global_failure or (failed and args.fail_fast):
            break

    print(
        "Batch inference: total={}, skipped={}, success={}, error={}, "
        "repaired_tail={}".format(
            len(items), skipped, succeeded, failed, str(repaired_tail).lower()
        ),
        file=sys.stderr,
    )
    if global_failure:
        return 3
    return 1 if failed else 0


def main(argv=None, *, runtime_factory=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args, runtime_factory=runtime_factory)
    except BatchInferenceError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
