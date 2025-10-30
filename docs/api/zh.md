---
layout: default
title: API 指南 (中文)
parent: API Documentation
nav_order: 2
---

# Emotion-LLaMA API 教程 (中文)
{: .no_toc }

在您的应用程序中使用 Emotion-LLaMA API 的完整指南。
{: .fs-6 .fw-300 }

[🇬🇧 Switch to English](en.md){: .btn .btn-blue }

---

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 环境配置

要运行 Emotion-LLaMA API，您需要创建一个新的 Conda 虚拟环境并安装特定版本的依赖：

### 版本要求

- **Python**: 3.10
- **Gradio**: 4.44.1
- **其他依赖**: 参考 `requirements.txt`

### 安装步骤

```bash
# 创建 Python 3.10 的 conda 环境
conda create -n llama_api python=3.10
conda activate llama_api

# 安装 Gradio 4.44.1
pip install gradio==4.44.1

# 安装其他依赖
pip install -r requirements.txt
```

---

## 部署客户端

在更新 Gradio 版本为 4.44.1 后，执行新的 Gradio 代码文件 `app_EmotionLlamaClient.py`。此代码简化了界面，只需要输入本地或者服务器的视频的绝对路径和 prompt，即可生成回复。

### 启动 API 服务器

```bash
python app_EmotionLlamaClient.py
```

{: .note }
> 注意 `app_EmotionLlamaClient.py` 代码中通过以下方式启动服务器：
> ```python
> iface.queue().launch(server_name="0.0.0.0", server_port=7889, share=False)
> ```
> 这将 Gradio API 设置为在 **7889 端口**运行。

API 可通过以下地址访问：
- **本地**: `http://127.0.0.1:7889`
- **网络**: `http://your-server-ip:7889`

---

## 使用 API

### 方案 1：使用 Python 发送请求

使用 Python 的 `requests` 模块发送 JSON 请求到 API。

#### 代码示例

```python
import json
import requests

# 记得修改 your-server-ip 为你的服务器 IP 或者 127.0.0.1
url = "http://your-server-ip:7889/api/predict/"
headers = {"Content-Type": "application/json"}

data = {
    "data": [
        "/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4",
        "The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

#### 响应格式

```json
{
  "data": [
    "视频中的人物显示出皱眉和紧张的面部表情（AU04 + AU07），表明沮丧或愤怒。声音语调尖锐且强调，音量提高。话语表达了对范小梅性格的不信和否认。这些多模态线索强烈反映了愤怒和沮丧的情绪。"
  ],
  "duration": 2.15
}
```

---

### 方案 2：使用 cURL

Gradio 的 API 默认在 `/api/predict/` 路径上，在 terminal 上直接执行：

#### Linux

```bash
# 注意不要使用单引号
curl -X POST "http://127.0.0.1:7889/api/predict/" \
     -H "Content-Type: application/json" \
     -d '{"data": ["/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4", "The person in video says: Won'\''t you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"]}'
```

#### Windows

```cmd
# Windows 环境下不能换行
curl -X POST "http://your-server-ip:7889/api/predict/" -H "Content-Type: application/json" -d "{\"data\": [\"/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4\", \"The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?\"]}"
```

---

### 方案 3：使用 FastAPI（推荐用于生产环境）

FastAPI 更高效，支持异步请求，非常适合生产环境使用。

#### 步骤 1：安装 FastAPI

首先，你要 pip 安装 FastAPI 和 Uvicorn：

```bash
pip install fastapi uvicorn
```

#### 步骤 2：创建 FastAPI 服务器

创建一个名为 `api_server.py` 的文件：

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

{: .warning }
> 注意，这里 FastAPI 的端口是 **7887**，不能与 Gradio 的端口 **7889** 冲突。

#### 步骤 3：启动 FastAPI 服务器

```bash
python api_server.py
```

API 可通过以下地址访问：
- **本地**: `http://127.0.0.1:7887`
- **网络**: `http://your-server-ip:7887`

#### 步骤 4：测试 API

**Linux:**

```bash
curl -X POST "http://your-server-ip:7887/process_video" \
     -H "Content-Type: application/json" \
     -d '{
          "video_path": "/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4",
          "question": "The person in video says: Won'\''t you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"
     }'
```

**Windows:**

```cmd
curl -X POST "http://your-server-ip:7887/process_video" -H "Content-Type: application/json" -d "{\"video_path\": \"/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4\", \"question\": \"The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?\"}"
```

**Python:**

```python
import requests

url = "http://your-server-ip:7887/process_video"
headers = {"Content-Type": "application/json"}

data = {
    "video_path": "/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4",
    "question": "The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"
}

response = requests.post(url, json=data)
print(response.json())
```

---

## API 接口说明

### Gradio API 端点

**URL**: `http://your-server-ip:7889/api/predict/`

**方法**: `POST`

**请求体**:
```json
{
  "data": [
    "video_path",
    "prompt_text"
  ]
}
```

**响应**:
```json
{
  "data": ["生成的响应文本"],
  "duration": 1.23
}
```

### FastAPI 端点

**URL**: `http://your-server-ip:7887/process_video`

**方法**: `POST`

**请求体**:
```json
{
  "video_path": "/path/to/video.mp4",
  "question": "您的问题或提示"
}
```

**响应**:
```json
{
  "response": "生成的响应文本"
}
```

**错误响应**:
```json
{
  "error": "错误信息描述"
}
```

---

## 常用提示词

### 情绪识别

```
[emotion] What is the emotion expressed in this video?
（这个视频表达了什么情绪？）
```

### 情绪推理

```
[reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind the words? Which emotion does this reflect?
（视频中使用了什么面部表情和声音语调？话语背后的意图是什么？这反映了哪种情绪？）
```

### 通用描述

```
Describe the person's emotional state in detail.
（详细描述这个人的情绪状态。）
```

---

## 最佳实践

### 视频路径

- 使用视频文件的**绝对路径**
- 确保 API 服务器对视频文件有**读取权限**
- 支持的格式：MP4、AVI、MOV

### 提示词工程

- 使用任务前缀：`[emotion]` 或 `[reason]`
- 明确说明您想要的信息
- 包含视频的上下文（例如，转录文本）

### 错误处理

始终实现适当的错误处理：

```python
try:
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
    response.raise_for_status()
    result = response.json()
    
    if "error" in result:
        print(f"API 错误: {result['error']}")
    else:
        print(f"结果: {result['data'][0]}")
        
except requests.exceptions.Timeout:
    print("请求超时")
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
```

---

## 性能考虑

### 响应时间

- **平均值**: 每个视频 1-3 秒
- **影响因素**: 视频长度、GPU 可用性、模型大小

### 并发请求

- Gradio API: 有限的并发性（基于队列）
- FastAPI: 更好的异步支持，适合多个请求

### 优化建议

1. **预处理视频**: 转换为最佳格式（MP4, H.264）
2. **限制视频长度**: 较短的视频（<30秒）处理更快
3. **使用缓存**: 为重复查询缓存结果
4. **负载均衡**: 在负载均衡器后使用多个 API 实例

---

## 安全注意事项

{: .warning }
> 默认 API 没有身份验证。对于生产使用：

1. **添加 API 密钥认证**:
   ```python
   @app.middleware("http")
   async def verify_api_key(request: Request, call_next):
       api_key = request.headers.get("X-API-Key")
       if api_key != "your-secret-key":
           return JSONResponse(status_code=401, content={"error": "无效的 API 密钥"})
       return await call_next(request)
   ```

2. **使用 HTTPS**: 加密 API 流量

3. **速率限制**: 防止滥用
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

4. **输入验证**: 验证视频路径和提示词

---

## 故障排除

### 常见问题

**问题**: 连接被拒绝
- **解决方案**: 确保 API 服务器正在运行且端口正确

**问题**: 超时错误
- **解决方案**: 增加超时值或检查 GPU 可用性

**问题**: "视频文件未找到"
- **解决方案**: 使用绝对路径并验证文件权限

**问题**: 响应时间慢
- **解决方案**: 使用 `nvidia-smi` 检查 GPU 利用率，减少视频长度

---

## 下一步

- 探索 [演示用法](../demo/) 获取交互式示例
- 查看 [主 API 文档](index.md) 获取概述
- 查看 [英文版本](en.md) (English version)

---

## 技术支持

如有 API 相关问题：
- 在 GitHub 上[提交问题](https://github.com/ZebangCheng/Emotion-LLaMA/issues)
- 查看[演示文档](../demo/)
- 查看[训练指南](../training/)

