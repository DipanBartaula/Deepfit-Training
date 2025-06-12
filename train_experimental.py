# train.py

import os
import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model import DeepFit
from utils import (
    JointVirtualTryOnDataset,
    seed_everything,
    setup_wandb,
    encode_prompt,
    prepare_control_input,
    setup_optimizer,
    save_checkpoint,
    load_checkpoint
)

import wandb  # for logging if enabled

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepFit (SD3-ControlNet) for Virtual Try-On (image-only latents)")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (if logging)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_experimental")
    parser.add_argument("--save_every_steps", type=int, default=100)
    parser.add_argument("--resume_step", type=int, default=None, help="If provided, resume from this step")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    return parser.parse_args()


def main():
    args = parse_args()
    # Set logging level to DEBUG if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)

    seed_everything(args.seed)

    # Setup W&B if requested
    if args.wandb_project:
        wandb_config = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "config": {
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "seed": args.seed
            },
            "tags": ["sd3", "controlnet", "virtual-tryon", "image-only-latents"]
        }
    else:
        wandb_config = None
    use_wandb = setup_wandb(wandb_config)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Dataset & DataLoader
    try:
        dataset = JointVirtualTryOnDataset(data_root=args.data_root)
    except NotImplementedError:
        logger.error("Dataset not implemented in utils.py. Please implement JointVirtualTryOnDataset.")
        return
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Dataset loaded. Number of samples: {len(dataset)}. Batch size: {args.batch_size}.")

    # Model: instantiate with 16-channel latents
    model = DeepFit(
        device=device,
        debug=args.debug,
        transformer_in_channels=16,
        transformer_out_channels=16,
        controlnet_in_latent_channels=16,
        controlnet_cond_channels=33  # typically unchanged: person(16) + mask(1) + clothing(16) = 33
    ).to(device)
    model.train()
    logger.info("Model instantiated (16-channel latents) and set to train mode.")

    # Optimizer
    optimizer = setup_optimizer(model, lr=args.lr)
    num_trainable = len([p for p in model.parameters() if p.requires_grad])
    logger.info(f"Optimizer set up with lr={args.lr}. Number of trainable params: {num_trainable}")

    # Resume if needed
    start_step = 0
    if args.resume_step is not None:
        try:
            model, optimizer = load_checkpoint(
                model, optimizer, args.resume_step,
                checkpoint_dir=args.checkpoint_dir, device=device
            )
            start_step = args.resume_step + 1
            logger.info(f"Resumed from checkpoint step {args.resume_step}. Starting at step {start_step}.")
        except Exception as e:
            logger.error(f"Error loading checkpoint at step {args.resume_step}: {e}")
            return

    global_step = start_step
    logger.info(f"Starting training from step {global_step}.")

    for epoch in range(args.num_epochs):
        epoch_losses = []
        logger.info(f"=== Epoch {epoch+1}/{args.num_epochs} ===")
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        for batch_idx, batch in pbar:
            # Move inputs to device; assume __getitem__ returns tensors [C,H,W]
            # We only need: person_image, mask, clothing_image, tryon_gt, and prompt
            person = batch["person_image"].to(device, dtype=torch.float16)      # [B, C, H, W]
            mask = batch["mask"].to(device, dtype=torch.float16)               # [B, 1, H, W]
            clothing = batch["clothing_image"].to(device, dtype=torch.float16) # [B, C, H, W]
            tryon_gt = batch["tryon_gt"].to(device, dtype=torch.float16)       # [B, C, H, W]
            prompts = batch["prompt"]  # list[str] length B

            B = person.shape[0]
            if args.debug:
                logger.debug(f"Batch {batch_idx+1}: person {person.shape}, mask {mask.shape}, clothing {clothing.shape}, tryon_gt {tryon_gt.shape}")

            # 1. Encode prompt
            prompt_embeds, pooled_prompt = encode_prompt(
                model=model,
                tokenizer1=model.tokenizer1,
                text_encoder1=model.text_encoder1,
                tokenizer2=model.tokenizer2,
                text_encoder2=model.text_encoder2,
                tokenizer3=model.tokenizer3,
                text_encoder3=model.text_encoder3,
                prompts=prompts,
                device=device,
                debug=args.debug
            )
            if args.debug:
                logger.debug(f"Encoded prompt: prompt_embeds shape {prompt_embeds.shape}, pooled_prompt shape {pooled_prompt.shape}")

            # 2. Prepare control input
            control_input = prepare_control_input(person, mask, clothing, vae=model.vae, debug=args.debug)
            if args.debug:
                logger.debug(f"Control input shape: {control_input.shape}")

            # 3. Encode tryon_gt via VAE to get image latents [B,16,h/8,w/8]
            latent_dist = model.vae.encode(tryon_gt).latent_dist
            tryon_latents = latent_dist.sample()
            tryon_latents = (tryon_latents - model.vae.config.shift_factor) * model.vae.config.scaling_factor
            if args.debug:
                logger.debug(f"Tryon image latents shape: {tryon_latents.shape}")

            # 4. Add noise ONLY to image latents
            timesteps = torch.rand(B, device=device)  # [B], continuous in [0,1]
            noise_img = torch.randn_like(tryon_latents)  # [B,16,h/8,w/8]
            noisy_img = tryon_latents + noise_img * timesteps[:, None, None, None]
            if args.debug:
                logger.debug(f"Noisy image latents shape: {noisy_img.shape}, timesteps shape: {timesteps.shape}")

            # 5. Forward through model: pass only image latents as hidden_states
            if args.debug:
                logger.debug("Starting model forward pass (image-only latents)...")
            model_pred = model(
                noisy_img,          # hidden_states: [B,16,h/8,w/8]
                timesteps,          # [B]
                control_input,      # conditioning
                prompt_embeds,
                pooled_prompt
            )
            # model_pred shape: [B,16,h/8,w/8]
            if args.debug:
                logger.debug(f"Model forward output shape: {model_pred.shape}")

            # 6. Compute loss on image channels: MSE between predicted noise and true noise_img
            # model_pred is predicted noise for image latents
            pred_img_noise = model_pred  # [B,16,...]
            loss = F.mse_loss(pred_img_noise.float(), noise_img.float())
            if args.debug:
                logger.debug(f"Computed loss (image-only) = {loss.item():.6f}")

            # 7. Backprop & gradient norm logging
            optimizer.zero_grad()
            loss.backward()
            if args.debug:
                logger.debug("Backward pass done. Computing gradient norms for all trainable parameters...")

            # Compute gradient norms
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    try:
                        norm = param.grad.norm().item()
                        grad_norms[f"grad_norm/{name}"] = norm
                    except Exception as e:
                        logger.warning(f"Could not compute grad norm for {name}: {e}")
            # Log to W&B
            if use_wandb and grad_norms:
                wandb.log(grad_norms, step=global_step)
            if args.debug and grad_norms:
                avg_norm = float(np.mean(list(grad_norms.values())))
                logger.debug(f"Average gradient norm: {avg_norm:.6f}")

            # 8. Optimizer step
            optimizer.step()
            if args.debug:
                logger.debug("Optimizer step completed.")

            # Record loss etc.
            epoch_losses.append(_
