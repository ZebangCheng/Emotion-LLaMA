### Project overview

Emotion-LLaMA is structured as follows:

```
📦Dataset
 ┗ 📦Emotion
    ┗ 📂MER2023
      ┣ 📂video
      ┣ 📂HL-UTT 
      ┣ 📂mae_340_UTT
      ┣ 📂maeV_399_UTT
      ┣ 📄transcription_en_all.csv
      ┣ 📄MERR_coarse_grained.txt
      ┣ 📄MERR_coarse_grained.json
      ┣ 📄MERR_fine_grained.txt
      ┗ 📄MERR_fine_grained.json
 📦Emotion-LLaMA
 ┣ 📂checkpoints
 ┃ ┣ 📂Llama-2-7b-chat-hf
 ┃ ┣ 📂save_checkpoint
 ┃ ┃ ┣ 📂stage2
 ┃ ┃ ┃ ┣ 🔖checkpoint_best.pth
 ┃ ┃ ┃ ┗ 📄log.txt
 ┃ ┃ ┗ 🔖Emoation_LLaMA.pth
 ┃ ┣ 📂transformer
 ┃ ┃ ┗ 📂chinese-hubert-large
 ┃ ┗ 🔖minigptv2_checkpoint.pth
 ┣ 📂eval_configs 
 ┃ ┣ 📜demo.yaml
 ┃ ┣ 📜eval_emotion.yaml
 ┃ ┗ 📜eval_emotion_EMER.yaml
 ┣ 📂train_configs
 ┃ ┣ 📜Emotion-LLaMA_finetune.yaml
 ┃ ┗ 📜minigptv2_tuning_stage_2.yaml
 ┣ 📂minigpt4
 ┣ 📑app.py
 ┣ 📜environment.yml
 ┣ 📑eval_emotion.py
 ┣ 📑eval_emotion_EMER.py
 ┗ 📑train.py

```