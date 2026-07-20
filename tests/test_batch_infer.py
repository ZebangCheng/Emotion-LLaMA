import contextlib
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).parents[1]
BATCH_SCRIPT = REPO_ROOT / "batch_infer.py"


def _load_batch_module():
    module_name = "batch_infer_under_test_{}".format(uuid.uuid4().hex)
    spec = importlib.util.spec_from_file_location(module_name, BATCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


class _FakeRuntime:
    def __init__(self, fail_names=()):
        self.fail_names = set(fail_names)
        self.load_calls = 0
        self.audio_load_calls = 0
        self.analyze_calls = []

    def load(self):
        self.load_calls += 1
        return self

    def load_audio_encoder(self):
        self.audio_load_calls += 1
        return self

    def analyze(self, video_path, prompt, *, seed=None, **generation_options):
        name = Path(video_path).name
        self.analyze_calls.append(
            {
                "name": name,
                "prompt": prompt,
                "seed": seed,
                "generation": generation_options,
            }
        )
        if name in self.fail_names:
            raise RuntimeError("inference failed for {}".format(name))
        return "answer for {}".format(name)


class BatchInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = _load_batch_module()

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = self.root / "demo.yaml"
        self.config.write_text("model: {}\n", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _manifest_argv(self, manifest, output, *extra):
        return [
            "--input",
            str(manifest),
            "--output",
            str(output),
            "--cfg-path",
            str(self.config),
            *extra,
        ]

    def _parse_manifest_args(self, manifest, output=None, *extra):
        if output is None:
            output = self.root / "unused.jsonl"
        return self.batch.build_parser().parse_args(
            self._manifest_argv(manifest, output, *extra)
        )

    def _run_main(self, argv, runtime_factory):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = self.batch.main(argv, runtime_factory=runtime_factory)
        return code, stderr.getvalue()

    @staticmethod
    def _read_output(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def _write_jsonl(path, records, trailing_newline=True):
        content = "\n".join(
            json.dumps(record, ensure_ascii=False) for record in records
        )
        if trailing_newline:
            content += "\n"
        path.write_text(content, encoding="utf-8")

    def _make_video_manifest(self, names):
        manifest = self.root / "input.jsonl"
        records = []
        for index, name in enumerate(names, start=1):
            (self.root / name).write_bytes(b"video")
            records.append(
                {
                    "id": "item-{}".format(index),
                    "video": name,
                    "prompt": "prompt {}".format(index),
                }
            )
        self._write_jsonl(manifest, records)
        return manifest

    def test_csv_and_jsonl_preserve_utf8_and_resolve_relative_paths(self):
        csv_directory = self.root / "csv 清单"
        csv_directory.mkdir()
        csv_video = csv_directory / "视频,一.mp4"
        csv_video.write_bytes(b"video")
        csv_manifest = csv_directory / "输入.csv"
        with csv_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("id", "video", "prompt"))
            writer.writeheader()
            writer.writerow(
                {
                    "id": "中文-id",
                    "video": csv_video.name,
                    "prompt": "先判断情绪，\n再解释原因",
                }
            )

        csv_items = self.batch.load_items(self._parse_manifest_args(csv_manifest))
        self.assertEqual(len(csv_items), 1)
        self.assertEqual(csv_items[0].item_id, "中文-id")
        self.assertEqual(csv_items[0].prompt, "先判断情绪，\n再解释原因")
        self.assertEqual(csv_items[0].resolved_video, csv_video.resolve())

        json_directory = self.root / "json 清单"
        media_directory = json_directory / "媒体"
        media_directory.mkdir(parents=True)
        json_video = media_directory / "片段二.mp4"
        json_video.write_bytes(b"video")
        json_manifest = json_directory / "输入.jsonl"
        json_record = {
            "id": "第二条",
            "video": "媒体/片段二.mp4",
            "prompt": "这个人是什么情绪？",
        }
        json_manifest.write_text(
            "\ufeff\n" + json.dumps(json_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        json_items = self.batch.load_items(self._parse_manifest_args(json_manifest))
        self.assertEqual(len(json_items), 1)
        self.assertEqual(json_items[0].item_id, "第二条")
        self.assertEqual(json_items[0].prompt, "这个人是什么情绪？")
        self.assertEqual(json_items[0].resolved_video, json_video.resolve())

    def test_single_video_and_directory_modes_have_stable_ids(self):
        video_directory = self.root / "videos with spaces"
        nested_directory = video_directory / "nested"
        nested_directory.mkdir(parents=True)
        upper_video = video_directory / "B.MP4"
        first_video = video_directory / "a.avi"
        nested_video = nested_directory / "c.mov"
        for video in (upper_video, first_video, nested_video):
            video.write_bytes(b"video")
        (video_directory / "ignored.txt").write_text("not a video", encoding="utf-8")

        parser = self.batch.build_parser()
        single_args = parser.parse_args(
            [
                "--video",
                str(upper_video),
                "--prompt",
                "single prompt",
                "--output",
                str(self.root / "single.jsonl"),
                "--cfg-path",
                str(self.config),
            ]
        )
        single_items = self.batch.load_items(single_args)
        self.assertEqual([item.item_id for item in single_items], ["B.MP4"])

        directory_args = parser.parse_args(
            [
                "--video-dir",
                str(video_directory),
                "--prompt",
                "directory prompt",
                "--output",
                str(self.root / "directory.jsonl"),
                "--cfg-path",
                str(self.config),
            ]
        )
        directory_items = self.batch.load_items(directory_args)
        self.assertEqual(
            [item.item_id for item in directory_items], ["a.avi", "B.MP4"]
        )

        recursive_args = parser.parse_args(
            [
                "--video-dir",
                str(video_directory),
                "--prompt",
                "directory prompt",
                "--output",
                str(self.root / "recursive.jsonl"),
                "--cfg-path",
                str(self.config),
                "--recursive",
            ]
        )
        recursive_items = self.batch.load_items(recursive_args)
        self.assertEqual(
            [item.item_id for item in recursive_items],
            ["a.avi", "B.MP4", "nested/c.mov"],
        )

    def test_success_failure_success_are_isolated_and_runtime_is_preloaded_once(self):
        manifest = self._make_video_manifest(("one.mp4", "two.mp4", "three.mp4"))
        output = self.root / "results.jsonl"
        runtime = _FakeRuntime(fail_names={"two.mp4"})
        factory_calls = []

        def runtime_factory():
            factory_calls.append(True)
            return runtime

        code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), runtime_factory
        )

        records = self._read_output(output)
        self.assertEqual(code, 1)
        self.assertEqual(factory_calls, [True])
        self.assertEqual(runtime.load_calls, 1)
        self.assertEqual(runtime.audio_load_calls, 1)
        self.assertEqual(
            [call["name"] for call in runtime.analyze_calls],
            ["one.mp4", "two.mp4", "three.mp4"],
        )
        self.assertEqual(
            [record["status"] for record in records],
            ["success", "error", "success"],
        )
        self.assertEqual(records[1]["error"]["stage"], "inference")
        self.assertEqual(records[1]["error"]["type"], "RuntimeError")
        self.assertEqual(
            set(records[0]),
            {
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
            },
        )

    def test_resume_skips_successes_and_retries_errors(self):
        manifest = self._make_video_manifest(("one.mp4", "two.mp4", "three.mp4"))
        output = self.root / "results.jsonl"
        first_runtime = _FakeRuntime(fail_names={"two.mp4"})
        first_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: first_runtime
        )
        self.assertEqual(first_code, 1)

        resumed_runtime = _FakeRuntime()
        second_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output, "--resume"),
            lambda: resumed_runtime,
        )

        records = self._read_output(output)
        self.assertEqual(second_code, 0)
        self.assertEqual(resumed_runtime.load_calls, 1)
        self.assertEqual(resumed_runtime.audio_load_calls, 1)
        self.assertEqual(
            [call["name"] for call in resumed_runtime.analyze_calls],
            ["two.mp4"],
        )
        self.assertEqual(
            {record["id"] for record in records},
            {"item-1", "item-2", "item-3"},
        )
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["status"] == "success" for record in records))

    def test_resume_rejects_a_changed_video_before_runtime_creation(self):
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        first_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: _FakeRuntime()
        )
        self.assertEqual(first_code, 0)
        original_output = output.read_bytes()

        video = self.root / "one.mp4"
        original_stat = video.stat()
        video.write_bytes(b"other")
        os.utime(
            video,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        self.assertEqual(video.stat().st_size, original_stat.st_size)
        self.assertEqual(video.stat().st_mtime_ns, original_stat.st_mtime_ns)
        factory_calls = []

        def forbidden_runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        resumed_code, stderr = self._run_main(
            self._manifest_argv(manifest, output, "--resume"),
            forbidden_runtime_factory,
        )

        self.assertEqual(resumed_code, 2)
        self.assertIn("video contents changed", stderr)
        self.assertEqual(factory_calls, [])
        self.assertEqual(output.read_bytes(), original_output)

    def test_same_metadata_video_change_during_inference_is_rejected(self):
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"

        class MutatingRuntime(_FakeRuntime):
            def analyze(inner_self, video_path, prompt, **kwargs):
                video = Path(video_path)
                original_stat = video.stat()
                video.write_bytes(b"other")
                os.utime(
                    video,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                return super(MutatingRuntime, inner_self).analyze(
                    video_path, prompt, **kwargs
                )

        code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: MutatingRuntime()
        )

        record = self._read_output(output)[0]
        self.assertEqual(code, 1)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"]["stage"], "validate")
        self.assertEqual(record["error"]["type"], "_VideoStateChangedError")

    def test_model_artifact_contents_are_part_of_run_fingerprint(self):
        checkpoint = self.root / "emotion.pth"
        llama_directory = self.root / "llama"
        audio_directory = self.root / "hubert"
        llama_directory.mkdir()
        audio_directory.mkdir()
        artifacts = (
            checkpoint,
            llama_directory / "model.bin",
            audio_directory / "pytorch_model.bin",
        )
        for artifact in artifacts:
            artifact.write_bytes(b"aaaa")

        self.config.write_text(
            json.dumps(
                {
                    "model": {
                        "ckpt": str(checkpoint),
                        "llama_model": str(llama_directory),
                        "audio_model_path": str(audio_directory),
                    }
                }
            ),
            encoding="utf-8",
        )
        manifest = self._make_video_manifest(("one.mp4",))
        args = self._parse_manifest_args(manifest)
        baseline = self.batch._run_fingerprint(args)

        for artifact in artifacts:
            with self.subTest(artifact=artifact.name):
                original_stat = artifact.stat()
                artifact.write_bytes(b"bbbb")
                os.utime(
                    artifact,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                self.assertEqual(artifact.stat().st_size, original_stat.st_size)
                self.assertEqual(
                    artifact.stat().st_mtime_ns, original_stat.st_mtime_ns
                )
                self.assertNotEqual(self.batch._run_fingerprint(args), baseline)
                artifact.write_bytes(b"aaaa")
                os.utime(
                    artifact,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )

    def test_resume_rejects_changed_model_artifact_before_runtime_creation(self):
        checkpoint = self.root / "emotion.pth"
        checkpoint.write_bytes(b"aaaa")
        self.config.write_text(
            json.dumps({"model": {"ckpt": str(checkpoint)}}),
            encoding="utf-8",
        )
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        first_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: _FakeRuntime()
        )
        self.assertEqual(first_code, 0)
        original_output = output.read_bytes()

        original_stat = checkpoint.stat()
        checkpoint.write_bytes(b"bbbb")
        os.utime(
            checkpoint,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        factory_calls = []

        def forbidden_runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        resumed_code, stderr = self._run_main(
            self._manifest_argv(manifest, output, "--resume"),
            forbidden_runtime_factory,
        )

        self.assertEqual(resumed_code, 2)
        self.assertIn("different runtime settings", stderr)
        self.assertEqual(factory_calls, [])
        self.assertEqual(output.read_bytes(), original_output)

    def test_shipped_default_llama_path_is_discovered_without_model_imports(self):
        args = self.batch.build_parser().parse_args(
            [
                "--video",
                str(self.root / "unused.mp4"),
                "--prompt",
                "prompt",
                "--output",
                str(self.root / "unused.jsonl"),
                "--cfg-path",
                str(REPO_ROOT / "eval_configs" / "demo.yaml"),
            ]
        )
        settings = self.batch._effective_model_settings(
            args,
            Path(args.cfg_path),
            REPO_ROOT,
        )

        self.assertEqual(
            settings["llama_model"], "checkpoints/Llama-2-7b-chat-hf"
        )
        self.assertEqual(
            settings["audio_model_path"],
            "checkpoints/transformer/chinese-hubert-large",
        )

        minimal_config = self.root / "minimal.yaml"
        minimal_config.write_text("model: {}\n", encoding="utf-8")
        minimal_settings = self.batch._effective_model_settings(
            args,
            minimal_config,
            REPO_ROOT,
        )
        self.assertEqual(
            minimal_settings["audio_model_path"],
            self.batch.DEFAULT_AUDIO_MODEL_PATH,
        )

    def test_model_artifact_options_override_configured_paths(self):
        manifest = self._make_video_manifest(("one.mp4",))
        overrides = {
            "ckpt": self.root / "override.pth",
            "llama_model": self.root / "override-llama",
            "audio_model_path": self.root / "override-hubert",
        }
        args = self.batch.build_parser().parse_args(
            self._manifest_argv(
                manifest,
                self.root / "unused.jsonl",
                "--options",
                *("model.{}={}".format(key, value) for key, value in overrides.items()),
            )
        )
        settings = self.batch._effective_model_settings(
            args, self.config, REPO_ROOT
        )

        for key, value in overrides.items():
            with self.subTest(key=key):
                self.assertEqual(settings[key], str(value))

    def test_model_selection_options_fail_closed(self):
        manifest = self._make_video_manifest(("one.mp4",))
        for index, option in enumerate(
            ("model.arch=minigpt_v2", "model.model_type=pretrain"), start=1
        ):
            with self.subTest(option=option):
                output = self.root / "selection-{}.jsonl".format(index)
                factory_calls = []

                def forbidden_runtime_factory():
                    factory_calls.append(True)
                    return _FakeRuntime()

                code, stderr = self._run_main(
                    self._manifest_argv(
                        manifest,
                        output,
                        "--options",
                        option,
                    ),
                    forbidden_runtime_factory,
                )

                self.assertEqual(code, 3)
                self.assertIn("is not supported for safe fingerprinting", stderr)
                self.assertEqual(factory_calls, [])
                self.assertFalse(output.exists())

    def test_interpolated_model_artifact_path_fails_closed(self):
        manifest = self._make_video_manifest(("one.mp4",))
        configurations = {
            "asset": (
                "asset_root: checkpoints\n"
                "model:\n"
                "  ckpt: ${asset_root}/emotion.pth\n"
            ),
            "architecture": (
                "arch_name: minigpt_v2\n"
                "model:\n"
                "  arch: ${arch_name}\n"
                "  model_type: pretrain\n"
            ),
            "model-node": (
                "selected_model:\n"
                "  arch: minigpt_v2\n"
                "  model_type: pretrain\n"
                "model: ${selected_model}\n"
            ),
            "dataset": (
                "model: {}\n"
                "datasets:\n"
                "  feature_face_caption:\n"
                "    vis_processor:\n"
                "      train:\n"
                "        image_size: ${oc.env:IMAGE_SIZE}\n"
            ),
        }

        for label, configuration in configurations.items():
            with self.subTest(label=label):
                self.config.write_text(configuration, encoding="utf-8")
                output = self.root / "results-{}.jsonl".format(label)
                factory_calls = []

                def forbidden_runtime_factory():
                    factory_calls.append(True)
                    return _FakeRuntime()

                code, stderr = self._run_main(
                    self._manifest_argv(manifest, output),
                    forbidden_runtime_factory,
                )

                self.assertEqual(code, 3)
                self.assertIn("inference configuration", stderr)
                self.assertEqual(factory_calls, [])
                self.assertFalse(output.exists())

        self.config.write_text("model: {}\n", encoding="utf-8")
        output = self.root / "results-option.jsonl"
        factory_calls = []

        def forbidden_option_runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        code, stderr = self._run_main(
            self._manifest_argv(
                manifest,
                output,
                "--options",
                "datasets.feature_face_caption.vis_processor.train.image_size="
                "${oc.env:IMAGE_SIZE}",
            ),
            forbidden_option_runtime_factory,
        )
        self.assertEqual(code, 3)
        self.assertIn("inference configuration", stderr)
        self.assertEqual(factory_calls, [])
        self.assertFalse(output.exists())

    def test_model_artifact_change_during_load_preserves_overwrite_target(self):
        checkpoint = self.root / "emotion.pth"
        checkpoint.write_bytes(b"aaaa")
        self.config.write_text(
            json.dumps({"model": {"ckpt": str(checkpoint)}}),
            encoding="utf-8",
        )
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        original_output = b"existing output\n"
        output.write_bytes(original_output)

        class MutatingRuntime(_FakeRuntime):
            def load(inner_self):
                checkpoint.write_bytes(b"bbbb")
                return super(MutatingRuntime, inner_self).load()

        code, stderr = self._run_main(
            self._manifest_argv(manifest, output, "--overwrite"),
            lambda: MutatingRuntime(),
        )

        self.assertEqual(code, 3)
        self.assertIn("changed while the runtime was loading", stderr)
        self.assertEqual(output.read_bytes(), original_output)

    def test_unreadable_video_is_a_validation_error_without_runtime(self):
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        factory_calls = []

        def forbidden_runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        with mock.patch.object(self.batch, "_video_fingerprint", return_value=None):
            code, _stderr = self._run_main(
                self._manifest_argv(manifest, output),
                forbidden_runtime_factory,
            )

        record = self._read_output(output)[0]
        self.assertEqual(code, 1)
        self.assertEqual(factory_calls, [])
        self.assertEqual(record["error"]["stage"], "validate")
        self.assertEqual(record["error"]["type"], "OSError")

    def test_resume_fail_fast_preserves_unattempted_error_records(self):
        manifest = self._make_video_manifest(("one.mp4", "two.mp4", "three.mp4"))
        output = self.root / "results.jsonl"
        first_runtime = _FakeRuntime(fail_names={"one.mp4", "two.mp4"})
        first_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: first_runtime
        )
        self.assertEqual(first_code, 1)

        resumed_runtime = _FakeRuntime(fail_names={"one.mp4"})
        resumed_code, _stderr = self._run_main(
            self._manifest_argv(
                manifest, output, "--resume", "--fail-fast"
            ),
            lambda: resumed_runtime,
        )

        records = self._read_output(output)
        self.assertEqual(resumed_code, 1)
        self.assertEqual(
            [call["name"] for call in resumed_runtime.analyze_calls], ["one.mp4"]
        )
        self.assertEqual(
            {record["id"]: record["status"] for record in records},
            {"item-1": "error", "item-2": "error", "item-3": "success"},
        )
        self.assertEqual(len(records), 3)

    def test_resume_normalizes_valid_record_without_final_newline(self):
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        runtime = _FakeRuntime()
        first_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: runtime
        )
        self.assertEqual(first_code, 0)

        output.write_bytes(output.read_bytes().rstrip(b"\n"))
        factory_calls = []

        def forbidden_runtime_factory():
            factory_calls.append(True)
            raise AssertionError("completed resume must not create a runtime")

        resumed_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output, "--resume"),
            forbidden_runtime_factory,
        )

        self.assertEqual(resumed_code, 0)
        self.assertEqual(factory_calls, [])
        self.assertTrue(output.read_bytes().endswith(b"\n"))
        self.assertEqual(len(self._read_output(output)), 1)

    def test_resume_discards_truncated_tail_and_reprocesses_that_item(self):
        manifest = self._make_video_manifest(("one.mp4", "two.mp4"))
        output = self.root / "results.jsonl"
        first_runtime = _FakeRuntime()
        first_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: first_runtime
        )
        self.assertEqual(first_code, 0)

        lines = output.read_bytes().splitlines(keepends=True)
        self.assertEqual(len(lines), 2)
        output.write_bytes(lines[0] + lines[1][:25].rstrip(b"\r\n"))

        resumed_runtime = _FakeRuntime()
        resumed_code, stderr = self._run_main(
            self._manifest_argv(manifest, output, "--resume"),
            lambda: resumed_runtime,
        )

        records = self._read_output(output)
        self.assertEqual(resumed_code, 0)
        self.assertIn("repaired_tail=true", stderr)
        self.assertEqual(
            [call["name"] for call in resumed_runtime.analyze_calls],
            ["two.mp4"],
        )
        self.assertEqual([record["id"] for record in records], ["item-1", "item-2"])
        self.assertTrue(all(record["status"] == "success" for record in records))

    def test_resume_rejects_schema_invalid_final_json_without_newline(self):
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        first_code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: _FakeRuntime()
        )
        self.assertEqual(first_code, 0)

        record = self._read_output(output)[0]
        record["schema_version"] = 999
        corrupted_output = json.dumps(record, ensure_ascii=False).encode("utf-8")
        output.write_bytes(corrupted_output)
        factory_calls = []

        def forbidden_runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        resumed_code, stderr = self._run_main(
            self._manifest_argv(manifest, output, "--resume"),
            forbidden_runtime_factory,
        )

        self.assertEqual(resumed_code, 4)
        self.assertIn("unsupported schema_version", stderr)
        self.assertEqual(factory_calls, [])
        self.assertEqual(output.read_bytes(), corrupted_output)

    def test_duplicate_ids_fail_before_runtime_or_output_creation(self):
        for name in ("one.mp4", "two.mp4"):
            (self.root / name).write_bytes(b"video")
        manifest = self.root / "duplicates.jsonl"
        self._write_jsonl(
            manifest,
            (
                {"id": "same", "video": "one.mp4", "prompt": "one"},
                {"id": "same", "video": "two.mp4", "prompt": "two"},
            ),
        )
        output = self.root / "results.jsonl"
        factory_calls = []

        def runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        code, stderr = self._run_main(
            self._manifest_argv(manifest, output), runtime_factory
        )

        self.assertEqual(code, 2)
        self.assertIn("Duplicate id", stderr)
        self.assertEqual(factory_calls, [])
        self.assertFalse(output.exists())

    def test_malformed_csv_and_overflow_rows_fail_closed(self):
        output = self.root / "results.jsonl"
        cases = {
            "unterminated.csv": 'video,prompt\na.mp4,"first prompt\nb.mp4,second prompt\n',
            "overflow.csv": "video,prompt\n,,orphan\nb.mp4,p\n",
        }
        for filename, content in cases.items():
            with self.subTest(filename=filename):
                manifest = self.root / filename
                manifest.write_text(content, encoding="utf-8")
                code, stderr = self._run_main(
                    self._manifest_argv(manifest, output), lambda: _FakeRuntime()
                )
                self.assertEqual(code, 2)
                self.assertIn("error:", stderr.casefold())
                self.assertFalse(output.exists())

    def test_duplicate_json_keys_are_rejected(self):
        manifest = self.root / "duplicate-key.jsonl"
        manifest.write_text(
            '{"id":"one","video":"first.mp4","video":"second.mp4",'
            '"prompt":"p"}\n',
            encoding="utf-8",
        )
        output = self.root / "results.jsonl"

        code, stderr = self._run_main(
            self._manifest_argv(manifest, output), lambda: _FakeRuntime()
        )

        self.assertEqual(code, 2)
        self.assertIn("duplicate key", stderr)
        self.assertFalse(output.exists())

    def test_locked_output_fails_before_runtime_creation(self):
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        factory_calls = []

        def runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        with self.batch._output_lock(output):
            code, stderr = self._run_main(
                self._manifest_argv(manifest, output), runtime_factory
            )

        self.assertEqual(code, 4)
        self.assertIn("output lock", stderr.casefold())
        self.assertEqual(factory_calls, [])
        self.assertFalse(output.exists())

    def test_runtime_initialization_failure_preserves_overwrite_target(self):
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        original = b"existing output must survive"
        output.write_bytes(original)

        def failing_runtime_factory():
            raise RuntimeError("model could not load")

        code, stderr = self._run_main(
            self._manifest_argv(manifest, output, "--overwrite"),
            failing_runtime_factory,
        )

        self.assertEqual(code, 3)
        self.assertIn("model could not load", stderr)
        self.assertEqual(output.read_bytes(), original)

    def test_all_missing_videos_write_validation_errors_without_runtime(self):
        manifest = self.root / "missing.jsonl"
        self._write_jsonl(
            manifest,
            (
                {"id": "missing-1", "video": "missing-one.mp4", "prompt": "one"},
                {"id": "missing-2", "video": "missing-two.mp4", "prompt": "two"},
            ),
        )
        output = self.root / "results.jsonl"
        factory_calls = []

        def runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        code, _stderr = self._run_main(
            self._manifest_argv(manifest, output), runtime_factory
        )

        records = self._read_output(output)
        self.assertEqual(code, 1)
        self.assertEqual(factory_calls, [])
        self.assertEqual(
            [record["status"] for record in records], ["error", "error"]
        )
        self.assertTrue(
            all(record["error"]["stage"] == "validate" for record in records)
        )

    def test_negative_seed_fails_before_runtime_creation(self):
        manifest = self._make_video_manifest(("one.mp4",))
        output = self.root / "results.jsonl"
        factory_calls = []

        def runtime_factory():
            factory_calls.append(True)
            return _FakeRuntime()

        code, stderr = self._run_main(
            self._manifest_argv(manifest, output, "--seed", "-1"),
            runtime_factory,
        )

        self.assertEqual(code, 2)
        self.assertIn("--seed must be non-negative", stderr)
        self.assertEqual(factory_calls, [])
        self.assertFalse(output.exists())


class BatchHelpTests(unittest.TestCase):
    def test_help_works_without_site_packages(self):
        completed = subprocess.run(
            [sys.executable, "-S", str(BATCH_SCRIPT), "--help"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--resume", completed.stdout)
        self.assertIn("--input", completed.stdout)


if __name__ == "__main__":
    unittest.main()
