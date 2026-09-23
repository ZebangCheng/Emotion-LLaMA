from collections.abc import Sequence
import json
import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


FEATURE_FACE_PATH_KEYS = (
    "transcription_path",
    "coarse_grained_json_path",
    "fine_grained_json_path",
    "face_feature_path",
    "video_feature_path",
    "audio_feature_path",
)
SUPPORTED_TASKS = ("emotion", "reason", "reason_v2")
ANNOTATION_FORMATS = ("auto", "ne", "ncev")
# Which single frame represents the clip: its first frame, its midpoint, or
# the AU-based emotional peak frame listed in peak_index_path.
FRAME_SELECTIONS = ("first", "middle", "peak")
DEFAULT_EMOTION_LABELS = (
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


def _resolve_dataset_path(path, base_dir):
    if path is None:
        return None
    path = os.fspath(path)
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normpath(path)


def feature_face_dataset_kwargs(dataset_config, path_config=None):
    path_config = dataset_config if path_config is None else path_config
    kwargs = {}
    task_pool = dataset_config.get("task_pool", None)
    if task_pool is not None:
        kwargs["task_pool"] = task_pool
    annotation_format = dataset_config.get("annotation_format", None)
    if annotation_format is not None:
        kwargs["annotation_format"] = annotation_format
    labels = dataset_config.get("labels", None)
    if labels is not None:
        kwargs["labels"] = labels
    prompt_labels = dataset_config.get("prompt_labels", None)
    if prompt_labels is not None:
        kwargs["prompt_labels"] = prompt_labels
    frame_selection = dataset_config.get("frame_selection", None)
    if frame_selection is not None:
        kwargs["frame_selection"] = frame_selection
    peak_index_path = path_config.get("peak_index_path", None)
    if peak_index_path is not None:
        kwargs["peak_index_path"] = peak_index_path
    for key in FEATURE_FACE_PATH_KEYS:
        value = path_config.get(key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _validate_task_pool(task_pool):
    if task_pool is None:
        return ["emotion"]
    error_message = (
        "Invalid task_pool {!r}; expected a non-empty list or tuple containing "
        "only supported tasks: {}"
    ).format(task_pool, ", ".join(SUPPORTED_TASKS))
    if (
        isinstance(task_pool, str)
        or not isinstance(task_pool, Sequence)
        or not task_pool
    ):
        raise ValueError(error_message)
    invalid = [task for task in task_pool if task not in SUPPORTED_TASKS]
    if invalid:
        raise ValueError(error_message)
    return list(task_pool)


def _validate_labels(labels):
    if labels is None:
        return list(DEFAULT_EMOTION_LABELS)
    if isinstance(labels, str) or not isinstance(labels, Sequence) or not labels:
        raise ValueError("labels must be a non-empty list or tuple")
    result = [str(label).strip() for label in labels]
    if any(not label for label in result):
        raise ValueError("labels cannot contain an empty value")
    if len(set(result)) != len(result):
        raise ValueError("labels must not contain duplicates")
    return result


def _get_transcript_sentence(character_lines, video_name):
    sentences = character_lines.loc[
        character_lines['name'] == video_name, 'sentence'
    ]
    if sentences.empty:
        raise KeyError("No transcript sentence found for sample {!r}".format(video_name))
    sentence = sentences.iloc[0]
    # A clip with no speech has an empty cell, which pandas reads as NaN.
    if pd.isna(sentence):
        return ""
    return sentence


def _validate_annotation_format(annotation_format):
    if annotation_format not in ANNOTATION_FORMATS:
        raise ValueError(
            "annotation_format must be one of {}".format(ANNOTATION_FORMATS)
        )


def _parse_annotation_line(line, line_number, annotation_format):
    _validate_annotation_format(annotation_format)
    fields = line.split()
    if annotation_format == "ne" or (
        annotation_format == "auto" and len(fields) == 2
    ):
        if len(fields) != 2:
            raise ValueError("Invalid NE annotation at line {}".format(line_number))
        return fields[0], fields[1]
    if annotation_format == "ncev" and len(fields) not in (3, 4):
        raise ValueError("Invalid NCEV annotation at line {}".format(line_number))
    if annotation_format == "auto" and len(fields) < 3:
        raise ValueError("Invalid NCEV annotation at line {}".format(line_number))
    return fields[0], fields[2]


class FeatureFaceDataset(Dataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_path,
        *,
        annotation_format="auto",
        task_pool=None,
        transcription_path=None,
        coarse_grained_json_path=None,
        fine_grained_json_path=None,
        face_feature_path="mae_340_UTT",
        video_feature_path="maeV_399_UTT",
        audio_feature_path="HL-UTT",
        labels=None,
        prompt_labels=None,
        frame_selection="first",
        peak_index_path=None,
        evaluation_mode=False,
        split="train",
    ):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.task_pool = _validate_task_pool(task_pool)
        self.labels = _validate_labels(labels)
        # The candidate list shown in the prompt. Under zero-shot transfer the
        # checkpoint's own label vocabulary may differ from the target
        # dataset's; scoring still uses self.labels.
        self.prompt_labels = (
            self.labels if prompt_labels is None else _validate_labels(prompt_labels)
        )
        if frame_selection not in FRAME_SELECTIONS:
            raise ValueError(
                "frame_selection must be one of {}".format(FRAME_SELECTIONS)
            )
        self.frame_selection = frame_selection
        self.peak_indexes = None
        self.evaluation_mode = bool(evaluation_mode)
        self.split = str(split)
        if self.evaluation_mode and len(set(self.task_pool)) != 1:
            raise ValueError(
                "evaluation_mode requires task_pool to contain exactly one task"
            )
        _validate_annotation_format(annotation_format)
        self.annotation_format = annotation_format

        self.caption_instruction_pool = [
            "Please describe the details of the expression and tone the video.",
            "Can you provide a description of the facial expression and tone shown by the person in the video?",
            "Could you outline the facial expressions and vocal tones displayed in the video?",
            "Detail the expressions and tone used in the video.",
            "Explain the visual and auditory expressions captured in the video.",
            "Provide an analysis of the expressions and tone featured in the video.",
        ]

        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: {}.".format(
                ", ".join(self.prompt_labels)
            ),

            # "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise.",
            # "Identify the displayed emotion in the video: is it happy, sad, neutral, angry, worried, or surprise?",
            # "Determine the emotional state shown in the video, choosing from happy, sad, neutral, angry, worried, or surprise.",
            # "Please ascertain the specific emotion portrayed in the video, whether it be happy, sad, neutral, angry, worried, or surprise.",
            # "Assess and label the emotion evident in the video: could it be happy, sad, neutral, angry, worried, surprise?",
        ]

        self.reason_instruction_pool = [
            "Please analyze all the clues in the video and reason out the emotional label of the person in the video.",
            "What is the emotional state of the person in the video? Please tell me the reason.",
            "What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?",
            "Please integrate information from various modalities to infer the emotional category of the person in the video.",
            "Could you describe the emotion-related features of the individual in the video? What emotional category do they fall into?",
        ]

        print("ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(os.path.abspath(ann_path))
        transcription_path = _resolve_dataset_path(transcription_path, self.file_path)
        coarse_grained_json_path = _resolve_dataset_path(
            coarse_grained_json_path, self.file_path
        )
        fine_grained_json_path = _resolve_dataset_path(
            fine_grained_json_path, self.file_path
        )
        self.face_feature_path = _resolve_dataset_path(
            face_feature_path, self.file_path
        )
        self.video_feature_path = _resolve_dataset_path(
            video_feature_path, self.file_path
        )
        self.audio_feature_path = _resolve_dataset_path(
            audio_feature_path, self.file_path
        )
        if self.frame_selection == "peak":
            if peak_index_path is None:
                raise ValueError(
                    "peak_index_path is required when frame_selection is 'peak'"
                )
            with open(
                _resolve_dataset_path(peak_index_path, self.file_path), "r"
            ) as peak_file:
                self.peak_indexes = json.load(peak_file)

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(self.labels): self.emo2idx[emo] = ii
        for ii, emo in enumerate(self.labels): self.idx2emo[ii] = emo

        self.samples = []
        with open(ann_path, encoding="utf-8") as annotation_file:
            for line_number, line in enumerate(annotation_file, start=1):
                if not line.strip():
                    continue
                video_name, emotion = _parse_annotation_line(
                    line, line_number, annotation_format
                )
                if emotion not in self.emo2idx:
                    raise ValueError(
                        "Unknown emotion label {!r} at line {}".format(
                            emotion, line_number
                        )
                    )
                self.samples.append((video_name, emotion))
        print(('video number:%d' % (len(self.samples))))

        self.MERR_coarse_grained_dict = None
        self.MERR_fine_grained_dict = None
        if "reason" in self.task_pool:
            if coarse_grained_json_path is None:
                raise ValueError(
                    "coarse_grained_json_path is required when task 'reason' is enabled"
                )
            with open(coarse_grained_json_path, 'r') as json_file:
                self.MERR_coarse_grained_dict = json.load(json_file)

        if "reason_v2" in self.task_pool:
            if fine_grained_json_path is None:
                raise ValueError(
                    "fine_grained_json_path is required when task 'reason_v2' is enabled"
                )
            with open(fine_grained_json_path, 'r') as json_file:
                self.MERR_fine_grained_dict = json.load(json_file)

        self.character_lines = None
        if transcription_path is not None:
            character_lines = pd.read_csv(transcription_path, dtype={"name": str})
            required_columns = ("name", "sentence")
            missing_columns = [
                column for column in required_columns if column not in character_lines.columns
            ]
            if missing_columns:
                raise ValueError(
                    "transcription_path is missing required column(s): {}".format(
                        ", ".join(missing_columns)
                    )
                )
            self.character_lines = character_lines


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_name, emotion_label = self.samples[index]

        video_path = os.path.join(self.vis_root, video_name + ".mp4")
        if os.path.exists(video_path):
            image = self.extract_frame(video_path, video_name)
        else:
            video_path = os.path.join(self.vis_root, video_name + ".avi")
            image = self.extract_frame(video_path, video_name)

        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)


        # image_file = '{}.jpg'.format(video_name)
        # image_path = os.path.join(self.vis_root, image_file)
        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)


        FaceMAE_feats, VideoMAE_feats, Audio_feats = self.get(video_name)
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)


        # random task
        task = self.task_pool[0] if self.evaluation_mode else random.choice(self.task_pool)
        if task == "emotion":
            target_raw = emotion_label
            caption = target_raw  # llama2 putput only emotion class
            caption = self.text_processor(caption)
            instruction_pool = self.emotion_instruction_pool
        elif task == "reason":
            target_raw = self.MERR_coarse_grained_dict[video_name]['caption']
            caption = target_raw

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool

        elif task == "reason_v2":
            target_raw = self.MERR_fine_grained_dict[video_name]['smp_reason_caption']
            caption = target_raw

            # caption = "" # for test reasoning

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool


        emotion = self.emo2idx[emotion_label]
        character_line = ""
        if self.character_lines is not None:
            sentence = _get_transcript_sentence(self.character_lines, video_name)
            character_line = "The person in video says: {}. ".format(sentence)
        
        instruction_template = (
            instruction_pool[0]
            if self.evaluation_mode
            else random.choice(instruction_pool)
        )
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, instruction_template)

        sample = {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": emotion,
            "image_id": video_name
        }
        if self.evaluation_mode:
            dataset_name = getattr(self, "name", "feature_face_caption")
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
    
    def extract_frame(self, video_path, video_name=None):
        video_capture = cv2.VideoCapture(video_path)
        target_index = None
        if self.frame_selection == "middle":
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 1:
                target_index = frame_count // 2
        elif self.frame_selection == "peak":
            entry = self.peak_indexes.get(str(video_name))
            if entry is None:
                raise KeyError(
                    "No peak index found for sample {!r}".format(video_name)
                )
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            target_index = int(entry["peak_index"])
            if frame_count > 0:
                target_index = min(target_index, frame_count - 1)
        if target_index:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        success, frame = video_capture.read()
        if not success and target_index:
            # Seeking can fail on a damaged index; fall back to the first frame.
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video_capture.read()
        if not success:
            raise ValueError("Failed to read video file:", video_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_capture.release()

        return frame_rgb


    def get(self, video_name):
        # FaceMAE feature
        FaceMAE_feats_path = os.path.join(self.face_feature_path, video_name + '.npy')
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature
        VideoMAE_feats_path = os.path.join(self.video_feature_path, video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature
        Audio_feats_path = os.path.join(self.audio_feature_path, video_name + '.npy')
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        return FaceMAE_feats, VideoMAE_feats, Audio_feats
