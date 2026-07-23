"""MER2024 dataset with configuration-driven labels and resource paths."""

from collections.abc import Sequence
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_MER2024_LABELS = (
    "neutral",
    "angry",
    "happy",
    "sad",
    "worried",
    "surprise",
    "fear",
    "contempt",
    "doubt",
)
MER2024_PATH_KEYS = (
    "transcription_path",
    "face_feature_path",
    "video_feature_path",
    "audio_feature_path",
)


def _resolve_dataset_path(path, base_dir):
    if path is None:
        return None
    path = os.fspath(path)
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normpath(path)


def _validate_labels(labels):
    if labels is None:
        return list(DEFAULT_MER2024_LABELS)
    if isinstance(labels, str) or not isinstance(labels, Sequence) or not labels:
        raise ValueError("labels must be a non-empty list or tuple")
    result = [str(label).strip() for label in labels]
    if any(not label for label in result):
        raise ValueError("labels cannot contain an empty value")
    if len(set(result)) != len(result):
        raise ValueError("labels must not contain duplicates")
    return result


def mer2024_dataset_kwargs(dataset_config, path_config=None):
    """Extract constructor options shared by builders and the offline CLI."""
    path_config = dataset_config if path_config is None else path_config
    kwargs = {}
    labels = dataset_config.get("labels", None)
    if labels is not None:
        kwargs["labels"] = labels
    for key in MER2024_PATH_KEYS:
        value = path_config.get(key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


class MER2024Dataset(Dataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_path,
        *,
        labels=None,
        transcription_path=None,
        face_feature_path="mae_340_23_UTT",
        video_feature_path="maeVideo_399_23_UTT",
        audio_feature_path="HL_23_UTT",
        evaluation_mode=False,
        split="train",
    ):
        self.vis_root = os.fspath(vis_root)
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.evaluation_mode = bool(evaluation_mode)
        self.split = str(split)
        self.task_pool = ["emotion"]
        self.labels = _validate_labels(labels)
        self.emos = self.labels
        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: {}.".format(
                ", ".join(self.labels)
            )
        ]

        self.ann_path = os.fspath(ann_path)
        self.file_path = os.path.dirname(os.path.abspath(self.ann_path))
        self.face_feature_path = _resolve_dataset_path(
            face_feature_path, self.file_path
        )
        self.video_feature_path = _resolve_dataset_path(
            video_feature_path, self.file_path
        )
        self.audio_feature_path = _resolve_dataset_path(
            audio_feature_path, self.file_path
        )

        self.tmp = []
        with open(self.ann_path, encoding="utf-8") as annotation_file:
            for line_number, line in enumerate(annotation_file, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) == 2:
                    emotion = fields[1]
                elif len(fields) >= 3:
                    emotion = fields[2]
                else:
                    raise ValueError(
                        "Invalid MER2024 annotation at line {}".format(line_number)
                    )
                if emotion not in self.labels:
                    raise ValueError(
                        "Unknown emotion label {!r} at line {}".format(
                            emotion, line_number
                        )
                    )
                self.tmp.append((fields[0], emotion))

        self.emo2idx = {
            emotion: index for index, emotion in enumerate(self.labels)
        }
        self.idx2emo = {
            index: emotion for index, emotion in enumerate(self.labels)
        }

        transcription_path = _resolve_dataset_path(
            transcription_path, self.file_path
        )
        self.character_lines = None
        self.transcription_column = None
        if transcription_path is not None:
            character_lines = pd.read_csv(transcription_path)
            if "name" not in character_lines.columns:
                raise ValueError("transcription_path is missing required column: name")
            for candidate in ("sentence_en", "sentence"):
                if candidate in character_lines.columns:
                    self.transcription_column = candidate
                    break
            if self.transcription_column is None:
                raise ValueError(
                    "transcription_path requires a sentence_en or sentence column"
                )
            self.character_lines = character_lines

    def __len__(self):
        return len(self.tmp)

    def _transcript_prefix(self, video_name):
        if self.character_lines is None:
            return ""
        sentences = self.character_lines.loc[
            self.character_lines["name"] == video_name,
            self.transcription_column,
        ]
        if sentences.empty:
            raise KeyError(
                "No transcript sentence found for sample {!r}".format(video_name)
            )
        return "The person in video says: {}. ".format(sentences.iloc[0])

    def __getitem__(self, index):
        video_name, emotion_label = self.tmp[index]

        video_path = os.path.join(self.vis_root, video_name + ".mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(self.vis_root, video_name + ".avi")
        image = self.extract_frame(video_path)
        image = Image.fromarray(image.astype("uint8")).convert("RGB")
        image = self.vis_processor(image)

        face_features, video_features, audio_features = self.get(video_name)
        feature_groups = (face_features, video_features, audio_features)
        feature_groups = tuple(
            feature.unsqueeze(0) if feature.ndim == 1 else feature
            for feature in feature_groups
        )
        combined_features = torch.cat(feature_groups, dim=0)

        target_raw = emotion_label
        caption = self.text_processor(target_raw)
        task = "emotion"
        instruction = (
            "<video><VideoHere></video> <feature><FeatureHere></feature> "
            "{} [{}] {} "
        ).format(
            self._transcript_prefix(video_name),
            task,
            self.emotion_instruction_pool[0],
        )

        sample = {
            "image": image,
            "video_features": combined_features,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": self.emo2idx[emotion_label],
            "image_id": video_name,
        }
        if self.evaluation_mode:
            dataset_name = getattr(self, "name", "mer2024_caption")
            sample.update(
                {
                    "dataset": dataset_name,
                    "split": self.split,
                    "task": task,
                    "sample_id": video_name,
                    "sample_index": index,
                    "instance_id": "{}:{}:{}:{}:{}".format(
                        dataset_name, self.split, index, task, video_name
                    ),
                    "target_raw": target_raw,
                }
            )
        return sample

    def extract_frame(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        success, frame = video_capture.read()
        if not success:
            video_capture.release()
            raise ValueError("Failed to read video file: {}".format(video_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_capture.release()
        return frame_rgb

    def get(self, video_name):
        feature_paths = (
            os.path.join(self.face_feature_path, video_name + ".npy"),
            os.path.join(self.video_feature_path, video_name + ".npy"),
            os.path.join(self.audio_feature_path, video_name + ".npy"),
        )
        return tuple(torch.tensor(np.load(path)) for path in feature_paths)
