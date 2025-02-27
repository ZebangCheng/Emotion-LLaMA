# Emotion-LLaMA API 教程
## 环境配置
你需要重新创建一个conda虚拟环境，python版本修改为3.10，gradio版本修改为 4.44.1，其他的包参考requirements.txt中的内容。
- python 3.10
- gradio 4.44.1

## 部署客户端
在更新gradio版本修改为 4.44.1版本后，执行新的gradio代码文件`app_EmotionLlamaClient.py`。此代码简化了界面，只需要输入本地或者服务器的视频的绝对路径和prompt，即可生成回复。
`python app_EmotionLlamaClient.py`

注意`app_EmotionLlamaClient.py`代码中通过`iface.queue().launch(server_name="0.0.0.0", server_port=7889, share=False)`设定gradio的端口为7889。

## 使用API
### 方案1：使用 Python 发送请求
用 Python requests 发送 JSON 请求，执行以下python代码
```python
import json
import requests

# 记得修改your-server-ip为你的服务器ip或者127.0.0.1
url = "http://your-server-ip:7889/api/predict/"
headers = {"Content-Type": "application/json"}

data = {
    "data": ["/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4",
             "The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

### 方案2：使用 curl 
Gradio 的 API 默认在 /api/predict/ 之类的路径上, 在teminal上直接执行
```
# linux：注意不要使用单引号 
curl -X POST "http://your-server-ip:7889/api/predict/" \
     -H "Content-Type: application/json" \
     -d '{"data": ["/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4", "The person in video says: Won'\''t you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"]}'

# windows：不能换行
curl -X POST "http://your-server-ip:7889/api/predict/" -H "Content-Type: application/json" -d "{\"data\": [\"/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4\", \"The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?\"]}"
```

### 方案3：使用 FastAPI
FastAPI 更高效，支持异步请求：
首先，你要pip安装FastAPI。

然后执行以下python代码：
```python
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from app_EmotionLlamaClient import process_video_question

app = FastAPI()

class VideoQuestionRequest(BaseModel):
    video_path: str
    question: str

@app.post("/process_video")
def process_video(req: VideoQuestionRequest):
    try:
        response = process_video_question(req.video_path, req.question)
        return {"response": response}
    except Exception as e:
        traceback.print_exc()  
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7887)
```

注意，这里FastAPI的端口是7887，不能与gradio的端口为7889冲突。

然后执行以下命令：
```
# for linux:
curl -X POST "http://your-server-ip:7887/process_video" \
     -H "Content-Type: application/json" \
     -d '{
          "video_path": "/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4",
          "question": "The person in video says: Won'\''t you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"
     }'

# for windows
curl -X POST "http://your-server-ip:7887/process_video" -H "Content-Type: application/json" -d "{\"video_path\": \"/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4\", \"question\": \"The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?\"}"
```
