# Multimodal emotion recognition and reasoning  

## Data source  

MER2023-SEMI contains over 70,000 unlabeled video clips. We utilize several powerful multimodal models to extract emotional cues from different modalities, and then use the latest LLaMA-3 model to summarize all emotional cues for inference, resulting in the final multimodal description.   

![MER2023_SEMI](./images/MER2023_SEMI.jpg)

## MERR Dataset Construction  

### 1. Data Filtering  

We employed OpenFace to extract faces from video segments, which were then aligned to identify various facial muscle movements, resulting in the detection of Action Units. Certain combinations of these muscle movements correlate with specific emotions. For instance, the emotion of surprise is identified through the combination of Action Unit 05 (upper lid raiser) and 26 (jaw drop). Each specific combination of Action Units was assigned a pseudo-label, signifying that the sample was selected and exhibited strong emotional expression characteristics. In total, 28,618 samples were selected and assigned pseudo-labels.

![au2label](./images/peak_frame_au_01.png)

### 2. Visual Expression Description  


### 3. Visual Objective Description  


### 4. Audio Tone Description  


### 5. Coarse-Grained Synthesis  


### 6. Fine-Grained Generation  






## Limitations