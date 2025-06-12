# model.py

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, SD3Transformer2DModel, SD3ControlNetModel, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from utils import modify_transformer_channels, modify_controlnet_channels, freeze_non_trainable_components


class DeepFit(nn.Module):
    """
    DeepFit model: loads VAE, text encoders, SD3 Transformer & ControlNet.
    Modifies Transformer and ControlNet channels to desired sizes, copies original weights and zero-inits new channels.
    Freezes VAE and text encoders; Transformer and ControlNet are fully trainable.
    """
    def __init__(self,
                 device: str = "cuda",
                 debug: bool = False,
                 transformer_in_channels: int = 20,
                 transformer_out_channels: int = 20,
                 controlnet_in_latent_channels: int = 20,
                 controlnet_cond_channels: int = 33):
        """
        Args:
            device: "cuda" or "cpu"
            debug: whether to print debug info
            transformer_in_channels: e.g. 20
            transformer_out_channels: e.g. 20
            controlnet_in_latent_channels: should equal transformer_out_channels
            controlnet_cond_channels: e.g. 33 (16 person + 1 mask + 16 clothing)
        """
        super().__init__()
        self.device = device
        self.debug = debug

        # 1. Load VAE
        if debug:
            print("[DEBUG] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="vae",
            torch_dtype=torch.float16
        ).to(device)
        if debug:
            print("[DEBUG] VAE loaded on", next(self.vae.parameters()).device)

        # 2. Load CLIP Text Encoder 1 & tokenizer
        if debug:
            print("[DEBUG] Loading CLIP Text Encoder 1...")
        self.tokenizer1 = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="tokenizer"
        )
        self.text_encoder1 = CLIPTextModelWithProjection.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="text_encoder",
            torch_dtype=torch.float16
        ).to(device)
        if debug:
            print("[DEBUG] CLIP Text Encoder 1 loaded on", next(self.text_encoder1.parameters()).device)

        # 3. Load CLIP Text Encoder 2 & tokenizer
        if debug:
            print("[DEBUG] Loading CLIP Text Encoder 2...")
        self.tokenizer2 = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="tokenizer_2"
        )
        self.text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="text_encoder_2",
            torch_dtype=torch.float16
        ).to(device)
        if debug:
            print("[DEBUG] CLIP Text Encoder 2 loaded on", next(self.text_encoder2.parameters()).device)

        # 4. Load T5 Encoder & tokenizer
        if debug:
            print("[DEBUG] Loading T5 Text Encoder...")
        self.tokenizer3 = T5TokenizerFast.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="tokenizer_3"
        )
        self.text_encoder3 = T5EncoderModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="text_encoder_3",
            torch_dtype=torch.float16
        ).to(device)
        if debug:
            print("[DEBUG] T5 Text Encoder loaded on", next(self.text_encoder3.parameters()).device)

        # 5. Load Transformer
        if debug:
            print("[DEBUG] Loading Transformer...")
        self.original_transformer = SD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="transformer",
            torch_dtype=torch.float16
        ).to(device)
        if debug:
            print("[DEBUG] Original Transformer loaded on", next(self.original_transformer.parameters()).device)

        # Modify Transformer channels
        if debug:
            print(f"[DEBUG] Modifying Transformer: in_channels={transformer_in_channels}, out_channels={transformer_out_channels}...")
        self.transformer = modify_transformer_channels(
            transformer=self.original_transformer,
            new_in_channels=transformer_in_channels,
            new_out_channels=transformer_out_channels,
            device=device
        )
        if debug:
            print("[DEBUG] Transformer modified.")

        # 6. Load ControlNet
        if debug:
            print("[DEBUG] Loading ControlNet...")
        self.controlnet_orig = SD3ControlNetModel.from_pretrained(
            "alimama-creative/SD3-Controlnet-Inpainting",
            use_safetensors=True,
            extra_conditioning_channels=1,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False
        ).to(device)
        if debug:
            print("[DEBUG] ControlNet loaded on", next(self.controlnet_orig.parameters()).device)

        # Modify ControlNet channels
        if debug:
            print(f"[DEBUG] Modifying ControlNet: latent_channels={controlnet_in_latent_channels}, cond_channels={controlnet_cond_channels}...")
        self.controlnet = modify_controlnet_channels(
            controlnet=self.controlnet_orig,
            in_channels_latent=controlnet_in_latent_channels,
            new_in_channels_cond=controlnet_cond_channels,
            device=device
        )
        if debug:
            print("[DEBUG] ControlNet modified.")

        # 7. Freeze non-trainable parts (VAE, text encoders); leave Transformer & ControlNet fully trainable
        freeze_non_trainable_components(self)
        if debug:
            print("[DEBUG] Frozen VAE and text encoders; Transformer & ControlNet trainable.")

        # 8. Scheduler for inference
        if debug:
            print("[DEBUG] Loading Scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="scheduler"
        )
        if debug:
            print("[DEBUG] Scheduler loaded.")

    def forward(self,
                noisy_latents: torch.Tensor,
                timesteps: torch.Tensor,
                control_input: torch.Tensor,
                prompt_embeds: torch.Tensor,
                pooled_prompt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: ControlNet then Transformer to predict noise (for training).
        noisy_latents: [B, latent_channels, h/8, w/8], float16
        timesteps: [B], float or int as required by scheduler/SD3
        control_input: [B, cond_channels, h/8, w/8], float16
        prompt_embeds: [B, seq_len, dim], float16
        pooled_prompt: [B, 2048], float16
        Returns model_pred [B, latent_channels, h/8, w/8], float16
        """
        control_block = self.controlnet(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt,
            controlnet_cond=control_input,
            conditioning_scale=1.0,
            return_dict=False,
        )[0]
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt,
            block_controlnet_hidden_states=control_block,
            return_dict=False,
        )[0]
        return model_pred
