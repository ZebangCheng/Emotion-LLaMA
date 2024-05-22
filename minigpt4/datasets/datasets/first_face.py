import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class FeatureFaceDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):

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
        self.file_path = os.path.dirname(ann_path)
        self.tmp = [x.strip().split(' ') for x in open(ann_path)]
        print(('video number:%d' % (len(self.tmp))))

        # emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']
        emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise', 'fear', 'contempt', 'doubt']

        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(emos): self.emo2idx[ii] = emo

        json_file_path = "/home/user/selected_face/face_emotion/AU_filter_merge.json"
        with open(json_file_path, 'r') as json_file:
            self.AU_filter_json = json.load(json_file)

        reason_json_file_path = "/home/user/selected_face/face_emotion/0512_target_smp_end.json"
        with open(reason_json_file_path, 'r') as json_file:
            self.reason_dict = json.load(json_file)

        self.character_lines = pd.read_csv('/home/user/selected_face/face_emotion/transcription_en_all.csv')


    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t[0]

        image_file = '{}.jpg'.format(video_name)
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

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
            caption = self.reason_dict[video_name]['smp_reason_caption']
            infer_str = " Therefore, it is inferred that his emotional state is: "
            caption = caption + infer_str + t[2]

            # caption = "" # for test reasoning

            caption = self.text_processor(caption)
            instruction_pool = self.reason_instruction_pool

        elif task == "infer":
            infer_str = " Therefore, it is inferred that his emotional state is: "
            caption = t[2]
            instruction_pool = [
                self.reason_dict[video_name]['reason_caption'] + infer_str,
            ]
        elif task == "caption":
            caption = self.AU_filter_json[video_name]['caption']
            caption = self.text_processor(caption)
            instruction_pool = self.caption_instruction_pool


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
    

    def get(self, video_name):
        # FaceMAE feature
        FaceMAE_feats_path = os.path.join(self.file_path, 'mae_340_UTT', video_name + '.npy')
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature
        VideoMAE_feats_path = os.path.join(self.file_path, 'maeV_399_UTT', video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature
        Audio_feats_path = os.path.join(self.file_path, 'HL-UTT', video_name + '.npy')
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        return FaceMAE_feats, VideoMAE_feats, Audio_feats
