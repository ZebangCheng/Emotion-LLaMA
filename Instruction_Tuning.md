# Instruction Tuning

## Setting Up the Pre-trained Model Parameter File

If you executed the Stage 1 pre-training code, the model parameter files will be saved in the `checkpoints/save_checkpoint` directory. Use the model parameter file from the last epoch and set the path in the configuration file:

```yaml
# Set Emotion-LLaMA path
ckpt: "/home/czb/project/Emotion-LLaMA/checkpoints/save_checkpoint/2024xxxx-v1/checkpoint_29.pth"
```

If you did not run the Stage 1 pre-training code, you can directly set the path to the demo model parameter file provided:

```yaml
# Set Emotion-LLaMA path
ckpt: "/home/user/project/Emotion-LLaMA/checkpoints/save_checkpoint/Emotion_LLaMA.pth"
```

## Setting Dataset Configuration

In the [dataset configuration file](minigpt4/configs/datasets/firstface/featureface.yaml#L11), select the use of `MERR_fine_grained.txt`. We use finer-grained data in Stage 2 to enhance Emotion-LLaMA’s emotional reasoning capabilities.

## Preparing Multi-task Instructions

First, define the types of tasks in the [dataset file](minigpt4/datasets/datasets/first_face.py#L61):

```python
self.task_pool = [
    # "emotion",
    # "reason",
    "reason_v2",
]
```

Next, retrieve the multimodal descriptions from the JSON file:

```python
caption = self.fine_grained_dict[video_name]['smp_reason_caption']

# caption = ""  # for test reasoning
```

## Running the Training

Run the following command to tune Emotion-LLaMA:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 train.py --cfg-path train_configs/minigptv2_tuning_stage_2.yaml
```

## Testing on the EMER Dataset

Specify the path to the Emotion-LLaMA checkpoint in the evaluation configuration file:

```yaml
# Set checkpoint path
llama_model: "/home/user/project/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf"
ckpt: "/home/user/project/Emotion-LLaMA/checkpoints/save_checkpoint/2024xxxx-v2/checkpoint_29.pth"
```

Please note that during the testing phase, set the caption to an empty string:

```python
# caption = self.fine_grained_dict[video_name]['smp_reason_caption']

caption = ""  # for test reasoning
```

Run the following command to evaluate the F1 score on MER2023-SEMI:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 eval_emotion_EMER.py --cfg-path eval_configs/eval_emotion_EMER.yaml
```

## Scoring

Finally, please refer to the code in the Affect open-source repository to score the predictions on the EMER dataset using ChatGPT.
> https://github.com/zeroQiaoba/AffectGPT/blob/master/AffectGPT/evaluation.py
