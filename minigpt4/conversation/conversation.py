import argparse
import os
import time
from threading import Thread
from PIL import Image
import cv2
from moviepy.editor import VideoFileClip
import soundfile as sf


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import Wav2Vec2FeatureExtractor

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from minigpt4.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop):] == stop).item():
                return True

        return False


CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

CONV_VISION_minigptv2 = Conversation(
    system="",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    else:
        print("Error: Cannot read frame from video.")
        return None

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    # audio.write_audiofile("audio.wav")

    audio_path = "audio.wav"
    audio.write_audiofile(audio_path, fps=16000, codec='pcm_s16le', ffmpeg_params=['-ac', '1'])
    samples, sr = sf.read(audio_path)
    return samples, sr

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0', stopping_criteria=None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer_prepare(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                       repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print('prompt:', prompt)
        # print('img_list:', img_list)
        embs = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs

    def answer(self, conv, img_list, **kargs):
        generation_dict = self.answer_prepare(conv, img_list, **kargs)
        output_token = self.model_generate(**generation_dict)[0]
        output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)

        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()

        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def stream_answer(self, conv, img_list, **kargs):
        print('stream_answer img shape: ', img_list[0].shape)
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        # print("Positional arguments (args):", args)
        # print("Keyword arguments (kwargs):", kwargs)
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output

    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)

        # # video
        if isinstance(image, str):  # is a video path
            print("isinstance str")
            video_path = image
            raw_image = get_first_frame(video_path)
            # cv2.imwrite("fisrt_frame.jpg", raw_image)
            frame_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            image = self.vis_processor(pil_image).unsqueeze(0).to(self.device)

            samples, sr = extract_audio_from_video(video_path)
            # print("samples:", samples)

            model_file = "checkpoints/transformer/chinese-hubert-large"
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)
            input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values
            # print("input_values:", input_values)

            from transformers import HubertModel
            hubert_model = HubertModel.from_pretrained(model_file)
            hubert_model.eval()
            with torch.no_grad():
                hidden_states = hubert_model(input_values, output_hidden_states=True).hidden_states # tuple of (B, T, D)
                # print("hidden_states:", hidden_states)
                audio_feature = torch.stack(hidden_states)[[-1]].sum(dim=0)  # sum, (B, T, D)
                audio_feature = audio_feature[0].detach().unsqueeze(0)
                audio_feature = torch.mean(audio_feature, dim=1, keepdim=True)

        elif isinstance(image, Image.Image):
            print("isinstance Image")
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            print("isinstance Tensor")
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        # print("audio_feature:", audio_feature)
        video_features = torch.zeros([1, 2, 1024])
        video_features = torch.cat((video_features, audio_feature), dim=1)

        print("audio faature shape:", audio_feature.shape)
        print("video_features", video_features.shape)
        image_emb, _ = self.model.encode_img(image, video_features)
        img_list.append(image_emb)

    def upload_img(self, image, conv, img_list):
        # conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        conv.append_message(conv.roles[0], "<video><VideoHere></video> <feature><FeatureHere></feature>")
        img_list.append(image)
        msg = "Received."

        return msg
    