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
    for key in FEATURE_FACE_PATH_KEYS:
        value = path_config.get(key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


class FeatureFaceDataset(Dataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_path,
        *,
        transcription_path="transcription_en_all.csv",
        coarse_grained_json_path="MERR_coarse_grained.json",
        fine_grained_json_path="MERR_fine_grained.json",
        face_feature_path="mae_340_UTT",
        video_feature_path="maeV_399_UTT",
        audio_feature_path="HL-UTT",
    ):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

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

        # self.task_pool = [
        #    "emotion",
        #    "reason",
        #    "infer",
        # ]

        self.task_pool = [
           "emotion",
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
        with open(ann_path) as annotation_file:
            self.tmp = [x.strip().split(' ') for x in annotation_file]
        print(('video number:%d' % (len(self.tmp))))

        # emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']
        emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise', 'fear', 'contempt', 'doubt']

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(emos): self.idx2emo[ii] = emo

        with open(coarse_grained_json_path, 'r') as json_file:
            self.MERR_coarse_grained_dict = json.load(json_file)

        with open(fine_grained_json_path, 'r') as json_file:
            self.MERR_fine_grained_dict = json.load(json_file)

        self.character_lines = pd.read_csv(transcription_path)


    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t[0]

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
            caption = t[2] # llama2 putput only emotion class
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


        emotion = self.emo2idx[t[2]]
        sentence = self.character_lines.loc[self.character_lines['name'] == video_name, 'sentence'].values[0]
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
