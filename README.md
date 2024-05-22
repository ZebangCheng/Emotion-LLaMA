# Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning  

## Introduction  

Emotion perception is crucial in applications like human-computer interaction, education, and counseling. 
However, traditional single-modality approaches fall short in real-world scenarios with multimodal emotional data, and Multimodal Large Language Models (MLLMs) struggle with audio integration and recognizing facial micro-expressions. 
To address this, we introduce the MERR dataset, containing 28,618 coarse-grained and 4,487 fine-grained annotated samples across diverse emotional categories.
This dataset enables models to learn from varied scenarios and generalize to real-world applications.
Furthermore, we propose Emotion-LLaMA, a model that integrates audio, visual, and textual inputs through emotion-specific encoders.
By aligning features into a shared space and employing a modified LLaMA model with instruction tuning, Emotion-LLaMA enhances both emotional recognition and reasoning capabilities.
Extensive evaluations show Emotion-LLaMA outperforms other MLLMs, achieving top scores in Clue Overlap (7.83) and Label Overlap (6.25) on EMER, an F1 score of 0.9036 on MER2023 challenge, and the highest UAR (45.59) and WAR (59.37) in zero-shot evaluations on DFEW dataset.

## Pipeline
![pipeline](./images/pipeline_prompt.jpg)
