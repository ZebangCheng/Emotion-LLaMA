import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.datasets.datasets.first_face import FeatureFaceDataset

# FeatureFaceDataset
@registry.register_builder("feature_face_caption")
class FirstfaceCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = FeatureFaceDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/firstface/featureface.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets 