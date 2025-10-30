---
layout: default
title: API Guide (English)
parent: API Documentation
nav_order: 1
---

# Emotion-LLaMA API Tutorial (English)
{: .no_toc }

Complete guide to using the Emotion-LLaMA API in your applications.
{: .fs-6 .fw-300 }

[🇨🇳 切换到中文](zh.md){: .btn .btn-blue }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Environment Setup

To run the Emotion-LLaMA API, you need to create a new Conda virtual environment with specific versions:

### Requirements

- **Python**: 3.10
- **Gradio**: 4.44.1
- **Other dependencies**: See `requirements.txt`

### Installation

```bash
# Create new conda environment with Python 3.10
conda create -n llama_api python=3.10
conda activate llama_api

# Install Gradio 4.44.1
pip install gradio==4.44.1

# Install other dependencies
pip install -r requirements.txt
```

---

## Deploying the Client

After updating Gradio to version 4.44.1, execute the `app_EmotionLlamaClient.py` script. This script simplifies the user interface, requiring only the absolute path of a local or server-based video file and a prompt to generate a response.

### Start the API Server

```bash
python app_EmotionLlamaClient.py
```

{: .note }
> In `app_EmotionLlamaClient.py`, the Gradio server is launched with:
> ```python
> iface.queue().launch(server_name="0.0.0.0", server_port=7889, share=False)
> ```
> This sets the Gradio API to run on **port 7889**.

The API will be accessible at:
- **Local**: `http://127.0.0.1:7889`
- **Network**: `http://your-server-ip:7889`

---

## Using the API

### Method 1: Python Requests

Use Python's `requests` module to send JSON requests to the API.

#### Code Example

```python
import json
import requests

# Replace "your-server-ip" with your server's IP address
# Or use 127.0.0.1 for local execution
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

#### Response Format

```json
{
  "data": [
    "The person in the video displays furrowed brows and a tense facial expression (AU04 + AU07), indicating frustration or anger. The vocal tone is sharp and emphatic, with raised volume. The words express disbelief and denial about Fan Xiaomei's character. These multimodal cues strongly reflect anger and frustration."
  ],
  "duration": 2.15
}
```

---

### Method 2: cURL Command

Gradio's API is available at `/api/predict/`. You can execute the following command in a terminal:

#### Linux

```bash
# Note: Avoid using single quotes for the video path
curl -X POST "http://127.0.0.1:7889/api/predict/" \
     -H "Content-Type: application/json" \
     -d '{"data": ["/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4", "The person in video says: Won'\''t you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"]}'
```

#### Windows

```cmd
# Single-line command required (no line breaks)
curl -X POST "http://your-server-ip:7889/api/predict/" -H "Content-Type: application/json" -d "{\"data\": [\"/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4\", \"The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?\"]}"
```

---

### Method 3: FastAPI (Recommended for Production)

FastAPI is more efficient and supports asynchronous requests, making it ideal for production environments.

#### Step 1: Install FastAPI

Ensure FastAPI and Uvicorn are installed:

```bash
pip install fastapi uvicorn
```

#### Step 2: Create FastAPI Server

Create a file named `api_server.py`:

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
> The FastAPI server runs on port **7887**, which should not conflict with the Gradio API running on port **7889**.

#### Step 3: Start the FastAPI Server

```bash
python api_server.py
```

The API will be available at:
- **Local**: `http://127.0.0.1:7887`
- **Network**: `http://your-server-ip:7887`

#### Step 4: Test the API

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

## API Endpoints

### Gradio API Endpoint

**URL**: `http://your-server-ip:7889/api/predict/`

**Method**: `POST`

**Request Body**:
```json
{
  "data": [
    "video_path",
    "prompt_text"
  ]
}
```

**Response**:
```json
{
  "data": ["Generated response text"],
  "duration": 1.23
}
```

### FastAPI Endpoint

**URL**: `http://your-server-ip:7887/process_video`

**Method**: `POST`

**Request Body**:
```json
{
  "video_path": "/path/to/video.mp4",
  "question": "Your question or prompt"
}
```

**Response**:
```json
{
  "response": "Generated response text"
}
```

**Error Response**:
```json
{
  "error": "Error message description"
}
```

---

## Common Prompts

### Emotion Recognition

```
[emotion] What is the emotion expressed in this video?
```

### Emotion Reasoning

```
[reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind the words? Which emotion does this reflect?
```

### General Description

```
Describe the person's emotional state in detail.
```

---

## Best Practices

### Video Path

- Use **absolute paths** for video files
- Ensure the API server has **read permissions** for the video file
- Supported formats: MP4, AVI, MOV

### Prompt Engineering

- Use task prefixes: `[emotion]` or `[reason]`
- Be specific about what information you want
- Include context from the video (e.g., transcription)

### Error Handling

Always implement proper error handling:

```python
try:
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
    response.raise_for_status()
    result = response.json()
    
    if "error" in result:
        print(f"API Error: {result['error']}")
    else:
        print(f"Result: {result['data'][0]}")
        
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

---

## Performance Considerations

### Response Time

- **Average**: 1-3 seconds per video
- **Factors**: Video length, GPU availability, model size

### Concurrent Requests

- Gradio API: Limited concurrency (queue-based)
- FastAPI: Better async support for multiple requests

### Optimization Tips

1. **Pre-process videos**: Convert to optimal format (MP4, H.264)
2. **Limit video length**: Shorter videos (<30s) process faster
3. **Use caching**: Cache results for repeated queries
4. **Load balancing**: Multiple API instances behind a load balancer

---

## Security Considerations

{: .warning }
> The default API has no authentication. For production use:

1. **Add API key authentication**:
   ```python
   @app.middleware("http")
   async def verify_api_key(request: Request, call_next):
       api_key = request.headers.get("X-API-Key")
       if api_key != "your-secret-key":
           return JSONResponse(status_code=401, content={"error": "Invalid API key"})
       return await call_next(request)
   ```

2. **Use HTTPS**: Encrypt API traffic

3. **Rate limiting**: Prevent abuse
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

4. **Input validation**: Validate video paths and prompts

---

## Troubleshooting

### Common Issues

**Issue**: Connection refused
- **Solution**: Ensure the API server is running and the port is correct

**Issue**: Timeout errors
- **Solution**: Increase timeout value or check GPU availability

**Issue**: "Video file not found"
- **Solution**: Use absolute paths and verify file permissions

**Issue**: Slow response times
- **Solution**: Check GPU utilization with `nvidia-smi`, reduce video length

---

## Next Steps

- Explore [demo usage](../demo/) for interactive examples
- Review [main API documentation](index.md) for overview
- Check the [Chinese version](zh.md) (中文版本)

---

## Support

For API-related questions:
- Open an [issue on GitHub](https://github.com/ZebangCheng/Emotion-LLaMA/issues)
- Review the [demo documentation](../demo/)
- Check [training guides](../training/)

