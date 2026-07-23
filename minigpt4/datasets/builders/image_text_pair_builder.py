import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.datasets.datasets.first_face import (
    FeatureFaceDataset,
    feature_face_dataset_kwargs,
)
from minigpt4.datasets.datasets.mer2024 import MER2024Dataset


SUPPORTED_SPLITS = ("train", "val", "test")


def annotation_paths_by_split(build_info):
    """Return explicit annotation paths, with legacy ``ann_path`` as train only."""
    annotations = build_info.get("annotations", None)
    if annotations is None:
        ann_path = build_info.get("ann_path", None)
        if ann_path is None:
            raise ValueError("build_info requires ann_path or annotations")
        return {"train": ann_path}
    if not hasattr(annotations, "items"):
        raise ValueError("build_info.annotations must map split names to paths")

    result = {}
    for split, ann_path in annotations.items():
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                "unsupported dataset split {!r}; expected train, val, or test".format(
                    split
                )
            )
        if ann_path is None or not str(ann_path).strip():
            raise ValueError("annotation path for split {!r} is empty".format(split))
        result[split] = ann_path
    if not result:
        raise ValueError("build_info.annotations cannot be empty")
    return result


def _processor_for_split(processors, split, processor_name):
    processor = processors.get("train" if split == "train" else "eval")
    if processor is None:
        raise ValueError(
            "{} processor is required for {} split".format(processor_name, split)
        )
    return processor


# FeatureFaceDataset
@registry.register_builder("feature_face_caption")
class FirstfaceCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = FeatureFaceDataset
    eval_dataset_cls = FeatureFaceDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/firstface/featureface.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        annotation_paths = annotation_paths_by_split(build_info)
        for split, ann_path in annotation_paths.items():
            is_train = split == "train"
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            dataset_kwargs = feature_face_dataset_kwargs(self.config, build_info)
            if not is_train:
                evaluation_task = self.config.get("evaluation_task", None)
                if evaluation_task is not None:
                    dataset_kwargs["task_pool"] = [str(evaluation_task)]
                dataset_kwargs.update({"evaluation_mode": True, "split": split})
            datasets[split] = dataset_cls(
                vis_processor=_processor_for_split(
                    self.vis_processors, split, "vis_processor"
                ),
                text_processor=_processor_for_split(
                    self.text_processors, split, "text_processor"
                ),
                ann_path=ann_path,
                vis_root=build_info.image_path,
                **dataset_kwargs,
            )

        return datasets
    
# MER2024Dataset
@registry.register_builder("mer2024_caption")
class MER2024nBuilder(BaseDatasetBuilder):
    train_dataset_cls = MER2024Dataset
    eval_dataset_cls = MER2024Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/firstface/mer2024.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        for split, ann_path in annotation_paths_by_split(build_info).items():
            is_train = split == "train"
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=_processor_for_split(
                    self.vis_processors, split, "vis_processor"
                ),
                text_processor=_processor_for_split(
                    self.text_processors, split, "text_processor"
                ),
                ann_path=ann_path,
                vis_root=build_info.image_path,
                evaluation_mode=not is_train,
                split=split,
            )

        return datasets
