# Emotion-LLaMA API Tutorial

## Environment Setup
To run the Emotion-LLaMA API, you need to create a new Conda virtual environment with Python 3.10 and install Gradio version 4.44.1. Other dependencies can be found in the `requirements.txt` file.

- Python 3.10
- Gradio 4.44.1

## Deploying the Client
After updating Gradio to version 4.44.1, execute the `app_EmotionLlamaClient.py` script. This script simplifies the user interface, requiring only the absolute path of a local or server-based video file and a prompt to generate a response.

```bash
python app_EmotionLlamaClient.py
```

Note that in `app_EmotionLlamaClient.py`, the Gradio server is launched with the following command:
```python
iface.queue().launch(server_name="0.0.0.0", server_port=7889, share=False)
```
This sets the Gradio API to run on port 7889.

## Using the API
### Method 1: Sending Requests with Python
You can use Python's `requests` module to send a JSON request. Run the following Python code:

```python
import json
import requests

# Replace "your-server-ip" with your server's IP address or use 127.0.0.1 for local execution.
url = "http://your-server-ip:7889/api/predict/"
headers = {"Content-Type": "application/json"}

data = {
    "data": ["/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4",
             "The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

### Method 2: Using cURL
Gradio’s API is available at `/api/predict/`. You can execute the following command in a terminal:

#### On Linux (avoid using single quotes):
```bash
curl -X POST "http://127.0.0.1:7889/api/predict/" \
     -H "Content-Type: application/json" \
     -d '{"data": ["/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4", "The person in video says: Won'\''t you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"]}'
```

#### On Windows (single-line command required):
```cmd
curl -X POST "http://your-server-ip:7889/api/predict/" -H "Content-Type: application/json" -d "{\"data\": [\"/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4\", \"The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?\"]}"
```

### Method 3: Using FastAPI
FastAPI is more efficient and supports asynchronous requests.

#### Step 1: Install FastAPI
Ensure FastAPI is installed using pip:
```bash
pip install fastapi uvicorn
```

#### Step 2: Run the Following Python Script
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

> **Note:** The FastAPI server runs on port **7887**, which should not conflict with the Gradio API running on port **7889**.

#### Step 3: Test the API with cURL

##### On Linux:
```bash
curl -X POST "http://your-server-ip:7887/process_video" \
     -H "Content-Type: application/json" \
     -d '{
          "video_path": "/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4",
          "question": "The person in video says: Won'\''t you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?"
     }'
```

##### On Windows:
```cmd
curl -X POST "http://your-server-ip:7887/process_video" -H "Content-Type: application/json" -d "{\"video_path\": \"/home/czb/project/Emotion-LLaMA/examples/sample_00004671.mp4\", \"question\": \"The person in video says: Won't you? Impossible! Fan Xiaomei is not such a person. [reason] What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?\"}"
```

