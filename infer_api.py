import os
import numpy as np
import cv2

import io
import base64
import requests
from PIL import Image
from io import BytesIO
import json
requests.adapters.DEFAULT_RETRIES = 5 # 增加重连次数


def get_pil_image_return_image_base64(raw_image_data):
    if raw_image_data is None:
        print("raw_image_data is None")
        return None
    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        if isinstance(raw_image_data["bytes"], bytes):
            return base64.b64encode(raw_image_data["bytes"]).decode('utf-8')
        else:
            raise ValueError("'bytes' key does not contain a bytes object")
    elif isinstance(raw_image_data, Image.Image):
        try:
            if raw_image_data.mode != "RGB":
                raw_image_data = raw_image_data.convert("RGB")
            buffered = io.BytesIO()
            raw_image_data.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            # to Base64 
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode PIL image: {str(e)}")
    elif isinstance(raw_image_data, str):
        return raw_image_data
    elif isinstance(raw_image_data, list):
        raw_image_data = raw_image_data[0]
        return get_pil_image_return_image_base64(raw_image_data)
    else:
        raise ValueError("Unsupported image data format")

        
temp_dir = "./temp"

video_path = "path/to/video.mp4"
# prompt = "分析视频中人物的情绪。"
# prompt = "What emotions does the video convey?"



if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


# 抽帧
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()  # 读取视频的第一帧
if not success:
    print(f"Failed to read video {video_name}.")

image_path = os.path.join(temp_dir, "temp_infer_EmotionLLaMA.jpg")
cv2.imwrite(image_path, frame)
        
url = "http://10.14.3.47:7889/api/predict/" #change this to your own url
headers = {"Content-Type": "application/json"}

raw_image_data = Image.open(image_path)

image = get_pil_image_return_image_base64(raw_image_data)
        
data = {
    "data": [image, prompt]
}

response = requests.post(url, headers=headers, data=json.dumps(data)).json()
print(response['data'][0])