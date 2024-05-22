import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel


@registry.register_model("minigpt_v2")
class MiniGPTv2(MiniGPTBase):
    """
    MiniGPT-v2 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_v2.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            llama_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        # lora_target_modules = ["q_proj", "v_proj"]
        # lora_r=128
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        img_f_dim = self.visual_encoder.num_features * 4
        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )
        
        self.feats_llama_proj1 = nn.Linear(
            1024, self.llama_model.config.hidden_size
        )
        self.feats_llama_proj2 = nn.Linear(
            1024, self.llama_model.config.hidden_size
        )
        self.feats_llama_proj3 = nn.Linear(
            1024, self.llama_model.config.hidden_size
        )
        
        self.cls_tk_llama_proj = nn.Linear(
            1408, self.llama_model.config.hidden_size
        )

        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()

    def encode_img(self, image, video_features):
        # device = 'cuda:0'
        device = image.device
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])
        with self.maybe_autocast():
            image_feats = self.visual_encoder(image)    # [1, 1025, 1408]
            image_embeds = self.ln_vision(image_feats).to(device)   # [1, 1025, 1408]
            image_cls_tk = image_embeds[:, :1, :]       # [1, 1, 1408]
            cls_tk_feats = self.cls_tk_llama_proj(image_cls_tk)    # [1, 1, 4096]
            image_embeds = image_embeds[:, 1:, :]       # [1, 1024, 1408]
            bs, pn, hs = image_embeds.shape             
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))  # [1, 256, 5632]
            image_inputs_llama = self.llama_proj(image_embeds)    # [1, 256, 4096]
            video_features = video_features.to(device)   # [1, 3, 1024]
            video_features_split = torch.split(video_features, 1, dim=1)
            output1 = self.feats_llama_proj1(video_features_split[0].squeeze(1))
            output2 = self.feats_llama_proj2(video_features_split[1].squeeze(1))
            output3 = self.feats_llama_proj3(video_features_split[2].squeeze(1))      
            video_feats = torch.stack([output1, output2, output3], dim=1)
            inputs_llama = torch.cat((image_inputs_llama, video_feats, cls_tk_feats), dim=1) # cls_tk_feats
            # inputs_llama = torch.cat((image_inputs_llama, video_feats), dim=1)

            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)    
        return inputs_llama, atts_llama

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Minigpt-4-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
