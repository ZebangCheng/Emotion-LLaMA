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


def _get_transcript_sentence(character_lines, video_name):
    sentences = character_lines.loc[
        character_lines['name'] == video_name, 'sentence'
    ]
    if sentences.empty:
        raise KeyError("No transcript sentence found for sample {!r}".format(video_name))
    return sentences.iloc[0]


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
    ):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.task_pool = _validate_task_pool(task_pool)
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
            "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.",

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
        # emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']
        emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise', 'fear', 'contempt', 'doubt']

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(emos): self.idx2emo[ii] = emo

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
            character_lines = pd.read_csv(transcription_path)
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
            image = self.extract_frame(video_path)
        else:
            video_path = os.path.join(self.vis_root, video_name + ".avi")
            image = self.extract_frame(video_path)

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
        task = random.choice(self.task_pool)
        if task == "emotion":
            caption = emotion_label  # llama2 putput only emotion class
            caption = self.text_processor(caption)
            instruction_pool = self.emotion_instruction_pool
        elif task == "reason":
            caption = self.MERR_coarse_grained_dict[video_name]['caption']

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool

        elif task == "reason_v2":
            caption = self.MERR_fine_grained_dict[video_name]['smp_reason_caption']

            # caption = "" # for test reasoning

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool


        emotion = self.emo2idx[emotion_label]
        character_line = ""
        if self.character_lines is not None:
            sentence = _get_transcript_sentence(self.character_lines, video_name)
            character_line = "The person in video says: {}. ".format(sentence)
        
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": emotion,
            "image_id": video_name
        }
    
    def extract_frame(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
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
