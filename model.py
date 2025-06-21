# model.py
"""
Defines DeepFit inpainting UNet for Virtual Try-On, loading pretrained Stable Diffusion v2 inpainting weights,
freezing all but self-attention parameters.
"""
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class DeepFit(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/stable-diffusion-2-inpainting",
        train_self_attention_only: bool = True,
    ):
        """
        Loads pretrained Stable Diffusion v2 inpainting UNet weights.
        If train_self_attention_only=True, freezes all params except those whose name includes 'attn' (case-insensitive).
        """
        super().__init__()
        print(f"[DeepFit] Loading UNet from pretrained '{pretrained_model_name}'...")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")
        print("[DeepFit] UNet loaded.")
        if train_self_attention_only:
            print("[DeepFit] Freezing all parameters except self-attention layers...")
            for name, param in self.unet.named_parameters():
                requires = "attn" in name.lower()
                param.requires_grad = requires
                if requires:
                    print(f"  [Trainable] {name}")
            print("[DeepFit] Freezing complete.")
        else:
            print("[DeepFit] All UNet parameters are trainable.")

    def forward(self, x, timesteps, encoder_hidden_states=None):
        # x: (B, 9, H, W) noisy latent + conditioning channels
        # timesteps: (B,) or scalar
        # encoder_hidden_states: (B, seq_len, embed_dim)
        return self.unet(x, timesteps=timesteps, encoder_hidden_states=encoder_hidden_states).sample
