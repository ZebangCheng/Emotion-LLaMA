"""Shared command-line evaluation pipeline used by legacy entry points."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.eval_utils import eval_parser
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.datasets.datasets.first_face import (
    FeatureFaceDataset,
    feature_face_dataset_kwargs,
)
from minigpt4.datasets.datasets.mer2024 import (
    MER2024Dataset,
    mer2024_dataset_kwargs,
)
from minigpt4.evaluation.evaluator import (
    csv_safe_value,
    evaluate_records,
    write_evaluation_report,
)
from minigpt4.evaluation.prompting import prepare_conversation_texts


SUPPORTED_DATASETS = ("feature_face_caption", "mer2024_caption", "dfew")

# Datasets whose samples are built by FeatureFaceDataset. DFEW reuses that
# loader: the only differences are the annotation file, the feature trees, and
# the label set, all of which come from the YAML config.
FEATURE_FACE_DATASETS = ("feature_face_caption", "dfew")


def comma_separated_values(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser(default_task="classification"):
    parser = eval_parser()
    parser.description = "Emotion-LLaMA evaluation"
    parser.add_argument(
        "--dataset",
        type=comma_separated_values,
        default=["feature_face_caption"],
        help="comma-separated evaluation dataset names",
    )
    parser.add_argument(
        "--task",
        choices=("auto", "classification", "reasoning"),
        default=default_task,
        help="metric family; legacy scripts select an appropriate default",
    )
    parser.add_argument(
        "--output-dir",
        help="artifact root; defaults to run.save_path from the YAML config",
    )
    parser.add_argument("--device", default="cuda:0", help="model device")
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help=(
            "evaluate a dataset the checkpoint was not fine-tuned on: records "
            "the run as zero-shot and applies the dataset's "
            "zero_shot_label_aliases when mapping generated text to labels"
        ),
    )
    parser.add_argument("--res", type=float, default=100.0, help=argparse.SUPPRESS)
    parser.add_argument("--resample", action="store_true", help=argparse.SUPPRESS)
    return parser


def _processor_from_config(dataset_cfg, kind):
    processor_group = dataset_cfg.get(kind)
    if processor_group is None:
        raise ValueError("dataset config is missing {}".format(kind))
    processor_cfg = processor_group.get("eval")
    if processor_cfg is None:
        processor_cfg = processor_group.get("train")
    if processor_cfg is None:
        raise ValueError("dataset config is missing {}.eval".format(kind))
    return registry.get_processor_class(processor_cfg.name).from_config(processor_cfg)


def _initialize_model(cfg, device):
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    return model


def _build_dataset(cfg, dataset_name, vis_processor, text_processor):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            "unsupported evaluation dataset {!r}; choose from {}".format(
                dataset_name, ", ".join(SUPPORTED_DATASETS)
            )
        )
    dataset_cfg = cfg.evaluation_datasets_cfg[dataset_name]
    if dataset_name in FEATURE_FACE_DATASETS:
        dataset = FeatureFaceDataset(
            vis_processor,
            text_processor,
            dataset_cfg["img_path"],
            dataset_cfg["eval_file_path"],
            evaluation_mode=True,
            split="test",
            **feature_face_dataset_kwargs(dataset_cfg),
        )
    elif dataset_name == "mer2024_caption":
        dataset = MER2024Dataset(
            vis_processor,
            text_processor,
            dataset_cfg["img_path"],
            dataset_cfg["eval_file_path"],
            evaluation_mode=True,
            split="test",
            **mer2024_dataset_kwargs(dataset_cfg),
        )
    dataset.name = dataset_name
    return dataset, dataset_cfg


def _as_list(value, size, default=None):
    if value is None:
        return [default] * size
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * size


def _generate_records(model, data_loader, dataset_name, dataset_cfg):
    conversation = CONV_VISION_minigptv2.copy()
    conversation.system = ""
    records = []
    next_index = 0
    max_new_tokens = int(dataset_cfg.get("max_new_tokens", 20))

    for batch in data_loader:
        instructions = batch["instruction_input"]
        texts = prepare_conversation_texts(instructions, conversation)
        predictions = model.generate(
            batch["image"],
            batch["video_features"],
            texts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        batch_size = len(predictions)
        sample_ids = _as_list(batch.get("sample_id", batch.get("image_id")), batch_size)
        sample_indexes = _as_list(batch.get("sample_index"), batch_size)
        instance_ids = _as_list(batch.get("instance_id"), batch_size)
        tasks = _as_list(batch.get("task"), batch_size, default="emotion")
        targets = _as_list(batch.get("target_raw", batch.get("answer")), batch_size)

        for offset, prediction in enumerate(predictions):
            sample_index = sample_indexes[offset]
            if sample_index is None:
                sample_index = next_index + offset
            sample_index = int(sample_index)
            sample_id = str(sample_ids[offset])
            task = str(tasks[offset])
            instance_id = instance_ids[offset]
            if instance_id is None:
                instance_id = "{}:test:{}:{}:{}".format(
                    dataset_name, sample_index, task, sample_id
                )
            records.append(
                {
                    "instance_id": str(instance_id),
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "split": "test",
                    "sample_index": sample_index,
                    "task": task,
                    "target": str(targets[offset]),
                    "prediction": str(prediction),
                }
            )
        next_index += batch_size
    return records


def _write_legacy_reasoning_csv(records, output_dir):
    path = Path(output_dir) / "output_Emotion-LLaMA.csv"
    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path.parent),
        prefix=".{}-".format(path.name),
        suffix=".tmp",
    )
    temporary_path = Path(temporary_file.name)
    try:
        with temporary_file:
            writer = csv.writer(temporary_file)
            writer.writerow(["names", "chi_reasons"])
            for record in records:
                writer.writerow(
                    [
                        csv_safe_value(record.get("sample_id", "")),
                        csv_safe_value(record["prediction"]),
                    ]
                )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(str(temporary_path), str(path))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return str(path)


def _evaluation_aliases(dataset_cfg, zero_shot):
    """Merge the dataset aliases with the zero-shot ones when requested."""
    aliases = dataset_cfg.get("label_aliases", None)
    if not zero_shot:
        return aliases
    zero_shot_aliases = dataset_cfg.get("zero_shot_label_aliases", None)
    if zero_shot_aliases is None:
        return aliases
    merged = dict(aliases or {})
    merged.update(dict(zero_shot_aliases))
    return merged


def run_dataset(model, cfg, dataset_name, task, output_root, zero_shot=False):
    training_dataset_cfg = cfg.datasets_cfg.get(dataset_name)
    if training_dataset_cfg is None:
        training_dataset_cfg = cfg.datasets_cfg[list(cfg.datasets_cfg.keys())[0]]
    vis_processor = _processor_from_config(training_dataset_cfg, "vis_processor")
    text_processor = _processor_from_config(training_dataset_cfg, "text_processor")
    dataset, dataset_cfg = _build_dataset(
        cfg, dataset_name, vis_processor, text_processor
    )
    data_loader = DataLoader(
        dataset,
        batch_size=int(dataset_cfg.get("batch_size", 1)),
        shuffle=False,
    )
    records = _generate_records(model, data_loader, dataset_name, dataset_cfg)
    labels = dataset_cfg.get("labels", getattr(dataset, "labels", None))
    aliases = _evaluation_aliases(dataset_cfg, zero_shot)
    report = evaluate_records(records, task=task, labels=labels, aliases=aliases)
    report["metrics"]["zero_shot"] = bool(zero_shot)
    output_dir = Path(output_root) / dataset_name
    paths = write_evaluation_report(report, output_dir)
    if report["metrics"]["task"] == "reasoning":
        paths["legacy_reasoning_csv"] = _write_legacy_reasoning_csv(
            report["records"], output_dir
        )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(paths, ensure_ascii=False, indent=2, sort_keys=True))
    return report, paths


def main(argv=None, default_task="classification"):
    parser = build_parser(default_task=default_task)
    args = parser.parse_args(argv)
    cfg = Config(args)
    output_root = args.output_dir or cfg.run_cfg.get("save_path", None)
    if not output_root:
        parser.error("set --output-dir or run.save_path in the YAML config")

    model = _initialize_model(cfg, args.device)
    for dataset_name in args.dataset:
        run_dataset(
            model,
            cfg,
            dataset_name,
            args.task,
            output_root,
            zero_shot=args.zero_shot,
        )
    return 0
