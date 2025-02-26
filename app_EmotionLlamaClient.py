import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import gradio as gr
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/demo.yaml', help="配置文件路径。")
    parser.add_argument(
        "--options",
        nargs="+",
        help="覆盖配置文件中的某些设置，格式为 xxx=yyy。",
    )
    args = parser.parse_args()
    return args

def load_model():
    args = parse_args()
    # args.instruct_ckpt = False # # bbb 2025年1月4日
    cfg = Config(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    
    vis_processor_cfg = cfg.datasets_cfg.feature_face_caption.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    model.eval()
    chat = Chat(model, vis_processor, device=device)
    
    return chat, device

chat, device = load_model()

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None

def process_video_question(video_path, question):
    if not os.path.exists(video_path):
        return "错误：视频文件不存在。"

    chat_state = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    # chat.upload_img(image, chat_state, [])
    # chat.upload_img(video_path, chat_state, [])
    chat_state.append_message(chat_state.roles[0], "<video><VideoHere></video> <feature><FeatureHere></feature>")
    img_list = []
    img_list.append(video_path)

    print('question: ', question)
    print('chat_state: ', chat_state)
    chat.ask(question, chat_state)

    print('img_list: ', img_list)
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)

    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        temperature=0.2,
        max_new_tokens=500,
        max_length=2000
    )[0]
    
    print('output:', response)
    return response

iface = gr.Interface(
    fn=process_video_question,
    inputs=[
        gr.Textbox(label="视频路径", placeholder="输入视频文件路径，例如：/path/to/video.mp4"),
        gr.Textbox(label="问题", placeholder="输入你的问题，例如：视频中的人物表达了什么情绪？")
    ],
    outputs=gr.Textbox(label="模型回答"),
    title="Emotion-LLaMA API",
    description="输入视频路径和问题，Emotion-LLaMA 将解析视频并回答你的问题。",
)

if __name__ == "__main__":
    # iface.launch(server_name="0.0.0.0", server_port=7889, share=True)
    # iface.launch(server_name="0.0.0.0", server_port=7889, share=False, enable_queue=True) # out
    iface.queue().launch(server_name="0.0.0.0", server_port=7889, share=False)


