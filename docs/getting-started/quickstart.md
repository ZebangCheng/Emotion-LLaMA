---
layout: default
title: Quick Start
parent: Getting Started
nav_order: 1
---

# Quick Start Guide
{: .no_toc }

Get up and running with Emotion-LLaMA in under 10 minutes.
{: .fs-6 .fw-300 }

---

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Before you begin, ensure you have:

- ✅ A system with NVIDIA GPU (24GB+ VRAM recommended)
- ✅ Conda or Miniconda installed
- ✅ Git installed
- ✅ At least 50GB free disk space

---

## 5-Minute Setup

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/ZebangCheng/Emotion-LLaMA.git
cd Emotion-LLaMA

# Create and activate conda environment
conda env create -f environment.yaml
conda activate llama
```

### Step 2: Download Pre-trained Models

Download the following models and place them in the `checkpoints/` directory:

1. **Llama-2-7b-chat-hf**: [Download from Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. **MiniGPT-v2**: Download the checkpoint (link in main README)
3. **Emotion-LLaMA Demo Model**: [Download from Google Drive](https://drive.google.com/file/d/1pNngqXdc3cKr9uLNW-Hu3SKvOpjzfzGY/view?usp=sharing)

### Step 3: Configure Paths

Update the model paths in the configuration files:

**In `minigpt4/configs/models/minigpt_v2.yaml`:**
```yaml
llama_model: "checkpoints/Llama-2-7b-chat-hf"
```

**In `eval_configs/demo.yaml`:**
```yaml
ckpt: "checkpoints/save_checkpoint/Emoation_LLaMA.pth"
```

### Step 4: Run the Demo

```bash
python app.py
```

Visit `http://127.0.0.1:7860` in your browser to try the demo!

---

## Try It Out

### Using the Web Interface

1. Upload a video file (or use one of the example videos from `examples/`)
2. Enter a prompt, such as:
   - "What emotion is expressed in this video?"
   - "Describe the facial expressions and tone."
   - "What is the person feeling and why?"
3. Click "Submit" to get the model's response

### Example Prompts

For **emotion recognition**:
```
[emotion] What is the emotion expressed in this video?
```

For **emotion reasoning**:
```
[reason] What are the facial expressions and vocal tone used in the video? 
What is the intended meaning behind the words? Which emotion does this reflect?
```

---

## Using the API

You can also use Emotion-LLaMA programmatically via the API.

### Python Example

```python
import json
import requests

url = "http://127.0.0.1:7889/api/predict/"
headers = {"Content-Type": "application/json"}

data = {
    "data": [
        "/path/to/video.mp4",
        "[emotion] What emotion is expressed in this video?"
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

For more API details, see the [API Documentation](../api/).

---

## What's Next?

Now that you have Emotion-LLaMA running, you can:

- 📊 [Explore the MERR Dataset](../dataset/) to understand the training data
- 🔧 [Train Your Own Model](../training/) with custom data
- 📈 [Run Evaluations](../evaluation/) to test performance
- 🔌 [Use the API](../api/) for integration into your applications

---

## Need Help?

- Check the [troubleshooting section](index.md#troubleshooting) for common issues
- Visit our [GitHub Issues](https://github.com/ZebangCheng/Emotion-LLaMA/issues) page
- Read the [full documentation](../)

Happy emotion recognition! 🎉

