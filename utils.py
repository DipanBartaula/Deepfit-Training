# utils.py

import os
import logging
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from diffusers import AutoencoderKL, SD3Transformer2DModel, SD3ControlNetModel, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image
import numpy as np

# Attempt to import PatchEmbed from diffusers; adjust if necessary
try:
    from diffusers.models.embeddings import PatchEmbed
except ImportError:
    from diffusers.models.embeddings import PatchEmbed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class JointVirtualTryOnDataset(Dataset):
    """
    A dummy dataset that returns random tensors *and* prompt embeddings for testing. 

    Each sample dict contains:
      - "person_image":    Tensor[3, H, W]
      - "mask":            Tensor[1, H, W]
      - "clothing_image":  Tensor[3, H, W]
      - "tryon_gt":        Tensor[3, H, W]
      - "depth_gt":        Tensor[1, H, W]
      - "normal_gt":       Tensor[3, H, W]
      - "prompt_embeds":   Tensor[B, seq_len, dim]
      - "pooled_prompt":   Tensor[B, 2048]
    """
    def __init__(
        self,
        data_root: Optional[str] = None,
        transform=None,
        num_samples: int = 1000,
        image_size: tuple = (1024, 1024),
        # new args for encoding
        tokenizer1: CLIPTokenizer = None,
        text_encoder1: CLIPTextModelWithProjection = None,
        tokenizer2: CLIPTokenizer = None,
        text_encoder2: CLIPTextModelWithProjection = None,
        tokenizer3: T5TokenizerFast = None,
        text_encoder3: T5EncoderModel = None,
        device: str = "cuda",
        debug: bool = False
    ):
        super().__init__()
        self.transform = transform
        self.num_samples = num_samples
        self.C, self.H, self.W = 3, *image_size

        # for prompt encoding
        assert all([tokenizer1, text_encoder1, tokenizer2, text_encoder2, tokenizer3, text_encoder3]), \
            "Must provide all three tokenizers and text encoders"
        self.tokenizer1 = tokenizer1
        self.text_encoder1 = text_encoder1
        self.tokenizer2 = tokenizer2
        self.text_encoder2 = text_encoder2
        self.tokenizer3 = tokenizer3
        self.text_encoder3 = text_encoder3
        self.device = device
        self.debug = debug

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # random log-uniform scale between 1e-3 and 1e3
        scale = 10 ** torch.empty(1).uniform_(-3, 3)

        # Base random Gaussian tensors
        person_image    = torch.randn(self.C, self.H, self.W) * scale
        mask            = torch.rand(1, self.H, self.W)         * scale
        clothing_image  = torch.randn(self.C, self.H, self.W) * scale
        tryon_gt        = torch.randn(self.C, self.H, self.W) * scale
        depth_gt        = torch.randn(1, self.H, self.W)     * scale
        normal_gt       = torch.randn(3, self.H, self.W)     * scale

        # Create a placeholder prompt
        prompt = f"sample prompt #{idx}"

        sample = {
            "person_image":   person_image,
            "mask":           mask,
            "clothing_image": clothing_image,
            "tryon_gt":       tryon_gt,
            "depth_gt":       depth_gt,
            "normal_gt":      normal_gt,
            "prompt":         prompt
        }

        # Apply any user-provided transform (e.g. normalization) to image tensors
        if self.transform is not None:
            sample = self.transform(sample)

        # Now encode the prompt into embeddings
        pe, pp = encode_prompt(
            model=None,  # not used inside encode_prompt
            tokenizer1=self.tokenizer1,
            text_encoder1=self.text_encoder1,
            tokenizer2=self.tokenizer2,
            text_encoder2=self.text_encoder2,
            tokenizer3=self.tokenizer3,
            text_encoder3=self.text_encoder3,
            prompts=[sample["prompt"]],
            device=self.device,
            debug=self.debug
        )
        # pe: [1, seq_len, dim], pp: [1, 2048]
        sample["prompt_embeds"] = pe.squeeze(0)
        sample["pooled_prompt"] = pp.squeeze(0)

        # remove raw prompt string if you like
        sample.pop("prompt")

        return sample


def modify_transformer_channels(transformer: SD3Transformer2DModel, new_in_channels: int, new_out_channels: int, device: str):
    """
    Replace Transformer PatchEmbed and proj_out to accept new_in_channels and output new_out_channels.
    Copies original weights into the first channels/features, zero-initializes new channels/features.
    Leaves entire weight tensors trainable (no gradient hooks).
    """
    # 1. Modify PatchEmbed (pos_embed)
    orig_patch_embed = transformer.pos_embed  # assumed PatchEmbed-like
    # Extract original weight/bias
    orig_weight = orig_patch_embed.proj.weight.data  # shape [embed_dim, in_orig, k, k]
    orig_bias = orig_patch_embed.proj.bias.data if hasattr(orig_patch_embed.proj, "bias") else None
    in_orig = orig_patch_embed.proj.in_channels
    if new_in_channels < in_orig:
        raise ValueError(f"new_in_channels ({new_in_channels}) < original in_channels ({in_orig})")
    embed_dim = orig_weight.shape[0]
    # Determine patch size and spatial info if needed
    # Many PatchEmbed constructors require height, width, patch_size, embed_dim, pos_embed_max_size.
    # We try to reuse existing attributes if present; otherwise rely on config.
    # For simplicity, assume the original PatchEmbed has attributes patch_size, patch_embed_size or can infer from config.
    # Adjust as needed if your diffusers version differs.
    if hasattr(orig_patch_embed, "patch_embed_size"):
        height, width = orig_patch_embed.patch_embed_size
    else:
        # fallback: if transformer.config contains image_size
        size = transformer.config.get("image_size", None)
        if size is None:
            raise ValueError("Cannot infer PatchEmbed image size; set patch_embed_size or config['image_size']")
        height = 128
        width = 96
    patch_size = 2
    # pos_embed_max_size = getattr(orig_patch_embed, "pos_embed_max_size", None)
    pos_embed_max_size = 96

    new_patch_embed = PatchEmbed(
        height=height,
        width=width,
        patch_size=patch_size,
        in_channels=new_in_channels,
        embed_dim=embed_dim,
        pos_embed_max_size=pos_embed_max_size
    ).to(device).to(torch.float16)

    # Copy original weights into first in_orig channels, zero-init the rest
    with torch.no_grad():
        new_patch_embed.proj.weight.data[:, :in_orig, :, :] = orig_weight
        if new_in_channels > in_orig:
            new_patch_embed.proj.weight.data[:, in_orig:, :, :].zero_()
        if orig_bias is not None:
            new_patch_embed.proj.bias.data[:] = orig_bias

    transformer.pos_embed = new_patch_embed
    logger.info(f"[DEBUG] Transformer PatchEmbed replaced: in_channels {in_orig} → {new_in_channels}")

    # 2. Modify final projection layer (proj_out)
    orig_proj_out = transformer.proj_out  # nn.Linear(in_features, out_features_orig)
    in_features = orig_proj_out.in_features
    out_orig = orig_proj_out.out_features
    # Compute new out_features: patch_size * patch_size * new_out_channels
    new_out_features = patch_size * patch_size * new_out_channels
    new_proj_out = nn.Linear(in_features, new_out_features).to(device).to(torch.float16)
    # Copy original weights/bias into first out_orig rows; zero-init new rows
    with torch.no_grad():
        # If new_out_features < out_orig, we truncate; but typically new_out_features >= out_orig
        rows_to_copy = min(out_orig, new_out_features)
        new_proj_out.weight.data[:rows_to_copy, :] = orig_proj_out.weight.data[:rows_to_copy, :]
        new_proj_out.bias.data[:rows_to_copy] = orig_proj_out.bias.data[:rows_to_copy]
        if new_out_features > out_orig:
            new_proj_out.weight.data[out_orig:, :].zero_()
            new_proj_out.bias.data[out_orig:].zero_()

    transformer.proj_out = new_proj_out
    # Update config/out_channels
    transformer.config["out_channels"] = new_out_channels
    transformer.out_channels = new_out_channels
    logger.info(f"[DEBUG] Transformer proj_out replaced: out_channels {out_orig // (patch_size*patch_size)} → {new_out_channels}")

    return transformer


def modify_controlnet_channels(controlnet: SD3ControlNetModel,
                               in_channels_latent: int,
                               new_in_channels_cond: int,
                               device: str):
    """
    Modify ControlNet pos_embed (for noisy_latents) to accept in_channels_latent,
    and pos_embed_input (conditioning) to accept new_in_channels_cond.
    Copies weights and zero-initializes new channels; leaves entire weight trainable.
    """
    # 1. Modify pos_embed for hidden_states (noisy_latents)
    orig_pos = controlnet.pos_embed
    orig_weight = orig_pos.proj.weight.data
    orig_bias = orig_pos.proj.bias.data if hasattr(orig_pos.proj, "bias") else None
    in_orig = orig_pos.proj.in_channels
    if in_channels_latent < in_orig:
        raise ValueError(f"in_channels_latent ({in_channels_latent}) < original ({in_orig})")
    embed_dim = orig_weight.shape[0]
    # Infer height/width and patch_size similar to transformer
    if hasattr(orig_pos, "patch_embed_size"):
        height, width = orig_pos.patch_embed_size
    else:
        size = 128
        if size is None:
            raise ValueError("Cannot infer ControlNet pos_embed image size; set patch_embed_size or config['image_size']")
        height =size
        width = 96
    patch_size = 2
    # pos_embed_max_size = getattr(orig_pos, "pos_embed_max_size", None)
    pos_embed_max_size=96

    new_pos_embed = PatchEmbed(
        height=height,
        width=width,
        patch_size=patch_size,
        in_channels=in_channels_latent,
        embed_dim=embed_dim,
        pos_embed_max_size=pos_embed_max_size
    ).to(device).to(torch.float16)

    with torch.no_grad():
        new_pos_embed.proj.weight.data[:, :in_orig, :, :] = orig_weight
        if in_channels_latent > in_orig:
            new_pos_embed.proj.weight.data[:, in_orig:, :, :].zero_()
        if orig_bias is not None:
            new_pos_embed.proj.bias.data[:] = orig_bias

    controlnet.pos_embed = new_pos_embed
    logger.info(f"[DEBUG] ControlNet pos_embed replaced: in_channels {in_orig} → {in_channels_latent}")

    # 2. Modify pos_embed_input (conditioning input)
    orig_pos_in = controlnet.pos_embed_input
    orig_weight_in = orig_pos_in.proj.weight.data
    orig_bias_in = orig_pos_in.proj.bias.data if hasattr(orig_pos_in.proj, "bias") else None
    in_orig_in = orig_pos_in.proj.in_channels
    if new_in_channels_cond < in_orig_in:
        raise ValueError(f"new_in_channels_cond ({new_in_channels_cond}) < original conditioning channels ({in_orig_in})")
    embed_dim_in = orig_weight_in.shape[0]
    if hasattr(orig_pos_in, "patch_embed_size"):
        height_in, width_in = orig_pos_in.patch_embed_size
    else:
        size = controlnet.config.get("image_size", None)
        if size is None:
            raise ValueError("Cannot infer ControlNet pos_embed_input image size; set patch_embed_size or config['image_size']")
        height_in =128
        width_in = 96
    patch_size_in = 2
    pos_embed_max_size_in = getattr(orig_pos_in, "pos_embed_max_size", None)

    new_pos_embed_in = PatchEmbed(
        height=height_in,
        width=width_in,
        patch_size=patch_size_in,
        in_channels=new_in_channels_cond,
        embed_dim=embed_dim_in,
        pos_embed_max_size=pos_embed_max_size_in
    ).to(device).to(torch.float16)

    with torch.no_grad():
        new_pos_embed_in.proj.weight.data[:, :in_orig_in, :, :] = orig_weight_in
        if new_in_channels_cond > in_orig_in:
            new_pos_embed_in.proj.weight.data[:, in_orig_in:, :, :].zero_()
        if orig_bias_in is not None:
            new_pos_embed_in.proj.bias.data[:] = orig_bias_in

    controlnet.pos_embed_input = new_pos_embed_in
    logger.info(f"[DEBUG] ControlNet pos_embed_input replaced: in_channels {in_orig_in} → {new_in_channels_cond}")

    return controlnet


def freeze_non_trainable_components(model: nn.Module):
    """
    Freeze VAE and text encoders; leave Transformer and ControlNet parameters trainable.
    Assumes model has attributes: .vae, .text_encoder1, .text_encoder2, .text_encoder3 (or similar).
    """
    # Freeze VAE parameters
    if hasattr(model, "vae"):
        for param in model.vae.parameters():
            param.requires_grad = False
    # Freeze text encoders
    for attr in ["text_encoder1", "text_encoder2", "text_encoder3"]:
        if hasattr(model, attr):
            enc = getattr(model, attr)
            for param in enc.parameters():
                param.requires_grad = False
    # Freeze tokenizers do not have parameters
    logger.info("[DEBUG] Frozen VAE and text encoder parameters. Transformer and ControlNet left trainable.")


def encode_prompt(model: nn.Module, tokenizer1: CLIPTokenizer, text_encoder1: CLIPTextModelWithProjection,
                  tokenizer2: CLIPTokenizer, text_encoder2: CLIPTextModelWithProjection,
                  tokenizer3: T5TokenizerFast, text_encoder3: T5EncoderModel,
                  prompts: list, device: str, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompts using CLIP1, CLIP2, and T5; combine into one embedding.
    Returns:
        prompt_embed: [B, seq_len_combined, dim_combined]
        pooled_prompt: dummy pooled prompt tensor [B, 2048]
    """
    # CLIP1
    tokens1 = tokenizer1(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    out1 = text_encoder1(tokens1, output_hidden_states=True)
    embed1 = out1.hidden_states[-2].to(torch.float16)  # [B, seq_len, dim1]

    # CLIP2
    tokens2 = tokenizer2(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    out2 = text_encoder2(tokens2, output_hidden_states=True)
    embed2 = out2.hidden_states[-2].to(torch.float16)  # [B, seq_len, dim1]

    # T5
    tokens3 = tokenizer3(prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt").input_ids.to(device)
    out3 = text_encoder3(tokens3)
    embed3 = out3.last_hidden_state.to(torch.float16)  # [B, seq_len_t5, dim3]

    # Combine CLIP1 + CLIP2 along last dim
    prompt_embed = torch.cat([embed1, embed2], dim=-1)  # [B, seq_len, 2*dim1]
    # Pad or truncate to match T5 dim
    t5_dim = embed3.shape[-1]
    clip_dim = prompt_embed.shape[-1]
    if clip_dim < t5_dim:
        prompt_embed = F.pad(prompt_embed, (0, t5_dim - clip_dim))
    elif clip_dim > t5_dim:
        prompt_embed = prompt_embed[:, :, :t5_dim]
        if debug:
            logger.warning(f"CLIP combined dim {clip_dim} > T5 dim {t5_dim}, truncating.")
    # Concatenate along sequence length
    prompt_embed = torch.cat([prompt_embed, embed3], dim=1)  # [B, seq_len+seq_len_t5, t5_dim]
    pooled_prompt = torch.zeros((len(prompts), 2048), device=device, dtype=torch.float16)  # dummy
    if debug:
        logger.debug(f"[DEBUG] Prompt embeddings shapes: {embed1.shape}, {embed2.shape}, {embed3.shape} -> combined {prompt_embed.shape}")
    return prompt_embed, pooled_prompt


def prepare_latents(batch_size: int, height: int, width: int, device: str, dtype=torch.float16) -> torch.Tensor:
    """
    Initialize random latents for inference.
    Output: [batch_size, 20, height//8, width//8]
    """
    shape = (batch_size, 20, height // 8, width // 8)
    latents = torch.randn(shape, device=device, dtype=dtype)
    logger.debug(f"[DEBUG] Initialized latents {latents.shape}")
    return latents


def encode_modality(vae: AutoencoderKL, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    Encode image x via VAE into latent space with scaling.
    x: [B, C, H, W]
    Returns [B, latent_C, H/8, W/8]
    """
    latent_dist = vae.encode(x).latent_dist
    latents = latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    if debug:
        logger.debug(f"[DEBUG] VAE Encoded shape: {latents.shape}")
    return latents


def prepare_control_input(person_images: torch.Tensor, masks: torch.Tensor, clothing_images: torch.Tensor,
                          vae: AutoencoderKL, debug: bool = False) -> torch.Tensor:
    """
    Encode person_images and clothing_images via VAE, resize mask, then concat → [B,33,h/8,w/8].
    """
    person_latents = encode_modality(vae, person_images, debug=debug)  # [B,16,h/8,w/8]
    clothing_latents = encode_modality(vae, clothing_images, debug=debug)  # [B,16,h/8,w/8]
    mask_latents = F.interpolate(masks, size=person_latents.shape[-2:], mode='nearest')  # [B,1,h/8,w/8]
    control_input = torch.cat([person_latents, mask_latents, clothing_latents], dim=1)  # [B,33,...]
    if debug:
        logger.debug(f"[DEBUG] Control input shape: {control_input.shape}")
    return control_input


def prepare_target_latents(tryon_gt: torch.Tensor, depth_gt: torch.Tensor, normal_gt: torch.Tensor,
                           vae: AutoencoderKL, debug: bool = False) -> torch.Tensor:
    """
    Encode tryon_gt via VAE, resize depth_gt and normal_gt, then concat → [B,20,h/8,w/8].
    """
    tryon_latents = encode_modality(vae, tryon_gt, debug=debug)  # [B,16,...]
    depth_latents = F.interpolate(depth_gt, size=tryon_latents.shape[-2:], mode='nearest')  # [B,1,...]
    normal_latents = F.interpolate(normal_gt, size=tryon_latents.shape[-2:], mode='nearest')  # [B,3,...]
    target_latents = torch.cat([tryon_latents, depth_latents, normal_latents], dim=1)  # [B,20,...]
    if debug:
        logger.debug(f"[DEBUG] Target latents shape: {target_latents.shape}")
    return target_latents


def add_noise(target_latents: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add noise in flow-matching style: continuous timesteps ∈ [0,1], noise, produce noisy_latents.
    Returns noisy_latents, noise, timesteps.
    """
    B = target_latents.shape[0]
    device = target_latents.device
    timesteps = torch.rand(B, device=device)  # ∈ [0,1]
    noise = torch.randn_like(target_latents)
    noisy_latents = target_latents + noise * timesteps[:, None, None, None]
    if debug:
        logger.debug(f"[DEBUG] Noisy latents shape {noisy_latents.shape}, timesteps {timesteps.shape}")
    return noisy_latents, noise, timesteps


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, checkpoint_dir: str = "checkpoints"):
    """
    Save model.state_dict and optimizer.state_dict to checkpoint_dir.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"deepfit_step_{step}.pth")
    optim_path = os.path.join(checkpoint_dir, f"optim_step_{step}.pth")
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optim_path)
    logger.info(f"[DEBUG] Saved checkpoint at step {step}: {model_path}, {optim_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int,
                    checkpoint_dir: str = "checkpoints", device: str = "cuda"):
    """
    Load model and optimizer state dicts for given step.
    """
    model_path = os.path.join(checkpoint_dir, f"deepfit_step_{step}.pth")
    optim_path = os.path.join(checkpoint_dir, f"optim_step_{step}.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model checkpoint at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    if os.path.isfile(optim_path):
        optimizer.load_state_dict(torch.load(optim_path, map_location=device))
        logger.info(f"[DEBUG] Loaded checkpoint step {step} into model and optimizer")
    else:
        logger.warning(f"No optimizer checkpoint at {optim_path}; loaded model only")
    return model, optimizer


def setup_optimizer(model: nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Set up AdamW optimizer including only parameters with requires_grad=True.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        logger.warning("No trainable parameters found for optimizer!")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    return optimizer


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeded everything with seed={seed}")


def setup_wandb(wandb_config: Dict[str, Any]):
    """
    Initialize W&B if wandb_config provided.
    """
    import wandb
    if wandb_config is not None:
        wandb.init(
            project=wandb_config.get("project", "sd3-virtual-tryon"),
            name=wandb_config.get("name", None),
            entity=wandb_config.get("entity", None),
            config=wandb_config.get("config", {}),
            tags=wandb_config.get("tags", [])
        )
        logger.info("[DEBUG] W&B initialized")
        return True
    else:
        logger.info("[DEBUG] No W&B logging")
        return False






def print_trainable_parameters(model, logger: logging.Logger = None):
    """
    Print (or log) all trainable parameters of the given model.

    Args:
        model: a torch.nn.Module whose trainable parameters we want to inspect.
        logger: optional logging.Logger. If provided, uses logger.info to output;
                otherwise, uses print().
    Returns:
        A list of tuples (name, parameter) for trainable parameters.
    """
    use_logger = logger is not None
    def _out(msg):
        if use_logger:
            logger.info(msg)
        else:
            print(msg)

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    _out(f"Trainable parameters ({len(trainable)} tensors):")
    total_params = 0
    for name, param in trainable:
        shape = tuple(param.shape)
        num = param.numel()
        total_params += num
        _out(f"  {name}: shape={shape}, params={num}")
    _out(f"Total trainable parameters: {total_params}")
    return trainable
