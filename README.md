# Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning  

## Introduction  

Emotion perception is crucial in applications like human-computer interaction, education, and counseling. 
However, traditional single-modality approaches fall short in real-world scenarios with multimodal emotional data, and Multimodal Large Language Models (MLLMs) struggle with audio integration and recognizing facial micro-expressions. 
To address this, we introduce the MERR dataset, containing 28,618 coarse-grained and 4,487 fine-grained annotated samples across diverse emotional categories.
This dataset enables models to learn from varied scenarios and generalize to real-world applications.
Furthermore, we propose Emotion-LLaMA, a model that integrates audio, visual, and textual inputs through emotion-specific encoders.
By aligning features into a shared space and employing a modified LLaMA model with instruction tuning, Emotion-LLaMA enhances both emotional recognition and reasoning capabilities.
Extensive evaluations show Emotion-LLaMA outperforms other MLLMs, achieving top scores in Clue Overlap (7.83) and Label Overlap (6.25) on EMER, an F1 score of 0.9036 on MER2023 challenge, and the highest UAR (45.59) and WAR (59.37) in zero-shot evaluations on DFEW dataset.

## MERR Dataset

### Comparison of Emotional Datasets
The MERR dataset extends the range of emotional categories and annotations beyond those found in existing datasets. Each sample is annotated with an emotion label and described in terms of its emotional expression.

![campare_datasets](./images/compare_datasets.jpg)

### Example of the MERR Dataset
The dataset was initially auto-annotated with coarse-grained labels for 28,618 samples from a large pool of unannotated data, and later refined to include 4,487 samples with fine-grained annotations. For more details on the data annotation process, see the [MERR](./MERR/README.md).

![example_sample_00000047](./images/data-example_sample_00000047_add_peak_00.png)


## Emotion-LLaMA
![Emotion-LLaMA](./images/framework.png)

## Setup
### Prepare the code and the environment

```
git clone https://github.com/ZebangCheng/Emotion-LLaMA.git
cd Emotion-LLaMA
conda env create -f environment.yaml
conda activate llama
```

### Prepare the pretrained LLM weights
Download the Llama-2-7b-chat-hf model from Huggingface to "Emotion-LLaMA/checkpoints/"  
```
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```
 
In the [model config file](minigpt4/configs/models/minigpt_v2.yaml#L14), specify the path to [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
```
# set Llama-2-7b-chat-hf path
llama_model: "/home/user/project/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf"
```


## Eval

### MER2023-SEMI
In the [evaluation config file](eval_configs/eval_emotion.yaml#L8), specify the path to [pretrained checkpoint of Emotion-LLaMA](https://github.com/ZebangCheng/Emotion-LLaMA/blob/main/README.md).  
```
# set pretrained checkpoint path
llama_model: "/home/user/project/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf"
ckpt: "/home/user/project/Emotion-LLaMA/checkpoints/save_checkpoint/stage2/checkpoint_best.pth"
```

Run the following code to evaluate the F1 score on MER2023-SEMI:  
```
torchrun  --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset feature_face_caption
```
2024-05-24

