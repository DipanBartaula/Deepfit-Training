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
    prepare_target_latents,
    add_noise,
    setup_optimizer,
    save_checkpoint,
    load_checkpoint
)

import wandb  # for logging if enabled

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepFit (SD3-ControlNet) for Virtual Try-On")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (if logging)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
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
            "tags": ["sd3", "controlnet", "virtual-tryon"]
        }
    else:
        wandb_config = None
    use_wandb = setup_wandb(wandb_config)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Dataset & DataLoader
    try:
        dataset = JointVirtualTryOnDataset(data_root=args.data_root)
    except NotImplementedError as e:
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

    # Model
    model = DeepFit(device=device, debug=args.debug).to(device)
    model.train()
    logger.info("Model instantiated and set to train mode.")

    # Optimizer
    optimizer = setup_optimizer(model, lr=args.lr)
    logger.info(f"Optimizer set up with lr={args.lr}. Number of trainable params: {len([p for p in model.parameters() if p.requires_grad])}")

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
            person = batch["person_image"].to(device, dtype=torch.float16)      # [B, C, H, W]
            mask = batch["mask"].to(device, dtype=torch.float16)               # [B, 1, H, W]
            clothing = batch["clothing_image"].to(device, dtype=torch.float16) # [B, C, H, W]
            tryon_gt = batch["tryon_gt"].to(device, dtype=torch.float16)       # [B, C, H, W]
            depth_gt = batch["depth_gt"].to(device, dtype=torch.float16)       # [B, 1, H, W]
            normal_gt = batch["normal_gt"].to(device, dtype=torch.float16)     # [B, 3, H, W]
            prompts = batch["prompt"]  # list[str] length B

            B = person.shape[0]
            if args.debug:
                logger.debug(f"Batch {batch_idx+1}: person {person.shape}, mask {mask.shape}, clothing {clothing.shape}, tryon_gt {tryon_gt.shape}, depth_gt {depth_gt.shape}, normal_gt {normal_gt.shape}")

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

            # 3. Prepare target latents
            target_latents = prepare_target_latents(tryon_gt, depth_gt, normal_gt, vae=model.vae, debug=args.debug)
            if args.debug:
                logger.debug(f"Target latents shape: {target_latents.shape}")

            # 4. Add noise
            noisy_latents, noise, timesteps = add_noise(target_latents, debug=args.debug)
            if args.debug:
                logger.debug(f"Noisy latents shape: {noisy_latents.shape}, noise shape: {noise.shape}, timesteps shape: {timesteps.shape}")

            # 5. Forward
            if args.debug:
                logger.debug("Starting model forward pass...")
            model_pred = model(noisy_latents, timesteps, control_input, prompt_embeds, pooled_prompt)
            if args.debug:
                logger.debug(f"Model forward output shape: {model_pred.shape}")

            # 6. Loss
            loss = F.mse_loss(model_pred.float(), noise.float())
            if args.debug:
                logger.debug(f"Computed loss: {loss.item():.6f}")

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
                    except Exception as e:
                        # In some rare cases, gradient might be sparse or unusual; skip if error
                        logger.warning(f"Could not compute grad norm for {name}: {e}")
                        continue
                    grad_norms[f"grad_norm/{name}"] = norm
            # Log to W&B
            if use_wandb and grad_norms:
                wandb.log(grad_norms, step=global_step)
            # Also print average grad norm
            if grad_norms:
                avg_norm = float(np.mean(list(grad_norms.values())))
                if args.debug:
                    logger.debug(f"Average gradient norm: {avg_norm:.6f}")

            # 8. Optimizer step
            optimizer.step()
            if args.debug:
                logger.debug("Optimizer step completed.")

            epoch_losses.append(loss.item())
            global_step += 1

            # Update progress bar description
            pbar.set_description(f"Step {global_step} Loss {loss.item():.4f}")

            # Log loss to W&B
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "train/step": global_step})

            # Save checkpoint periodically
            if global_step % args.save_every_steps == 0:
                logger.info(f"Saving checkpoint at step {global_step}...")
                save_checkpoint(model, optimizer, global_step, checkpoint_dir=args.checkpoint_dir)

        # End of epoch
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} completed. Avg Loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({"epoch/loss": avg_loss, "epoch": epoch+1}, step=global_step)
        # Save at end of epoch
        logger.info(f"Saving checkpoint at end of epoch {epoch+1}, step {global_step}...")
        save_checkpoint(model, optimizer, global_step, checkpoint_dir=args.checkpoint_dir)

    if use_wandb:
        wandb.finish()
        logger.info("W&B run finished.")


if __name__ == "__main__":
    main()
