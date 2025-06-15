# train.py

import os
import argparse
import logging
import torch
import torch.nn.functional as F
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
    load_checkpoint,
    print_trainable_parameters
)

from virtual_try_on_dataloader import get_train_val_loaders

import wandb  # for logging if enabled

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepFit (SD3-ControlNet) for Virtual Try-On")
    p.add_argument("--data_root", type=str, default=None,
                  help="Path to data (omit for dummy dataset)")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--batch_size", type=int, default=1, help="per-step batch size")
    p.add_argument("--effective_batch_size", type=int, default=128,
                  help="total effective batch size via gradient accumulation")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every_steps", type=int, default=100)
    p.add_argument("--resume_step", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    seed_everything(args.seed)

    # Gradient accumulation
    if args.effective_batch_size % args.batch_size != 0:
        raise ValueError("effective_batch_size must be multiple of batch_size")
    accum_steps = args.effective_batch_size // args.batch_size
    logger.info(f"Accumulating over {accum_steps} steps to reach batch {args.effective_batch_size}")

    # W&B setup
    wandb_cfg = None
    if args.wandb_project:
        wandb_cfg = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "config": {
                "batch_size": args.batch_size,
                "effective_batch_size": args.effective_batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "seed": args.seed
            },
            "tags": ["sd3", "controlnet", "virtual-tryon"]
        }
    use_wandb = setup_wandb(wandb_cfg)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # DataLoaders
    if args.data_root:
        train_loader, val_loader = get_train_val_loaders(
            args.data_root,
            batch_size=args.batch_size,
            val_fraction=0.1,
            seed=args.seed,
            num_workers=4
        )
    else:
        dummy = JointVirtualTryOnDataset()
        train_loader = torch.utils.data.DataLoader(dummy, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True)
        val_loader   = torch.utils.data.DataLoader(dummy, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=2, pin_memory=True)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model & optimizer
    model = DeepFit(device=device, debug=args.debug).to(device)
    model.train()
    optimizer = setup_optimizer(model, lr=args.lr)
    trainable=print_trainable_parameters(model)
    logger.info(f"Trainable parameters: {trainable}")

    # Resume?
    start_step = 0
    if args.resume_step is not None:
        model, optimizer = load_checkpoint(
            model, optimizer, args.resume_step,
            checkpoint_dir=args.checkpoint_dir, device=device
        )
        start_step = args.resume_step + 1
    global_step = start_step
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        # — Training —
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # to device
            person   = batch["person_image"].to(device, dtype=torch.float16)
            mask     = batch["mask"].to(device, dtype=torch.float16)
            cloth    = batch["cloth_image"].to(device, dtype=torch.float16)
            tryon    = batch["overlay_image"].to(device, dtype=torch.float16)
            depth    = batch["depth_map"].to(device, dtype=torch.float16)
            normal   = batch["normal_map"].to(device, dtype=torch.float16)
            prompts  = batch["caption"]

            # forward
            pe, pp = encode_prompt(model, model.tokenizer1, model.text_encoder1,
                                   model.tokenizer2, model.text_encoder2,
                                   model.tokenizer3, model.text_encoder3,
                                   prompts, device, args.debug)
            ctrl = prepare_control_input(person, mask, cloth, model.vae, args.debug)
            tgt  = prepare_target_latents(tryon, depth, normal, model.vae, args.debug)
            noised, noise, t = add_noise(tgt, args.debug)
            pred = model(noised, t, ctrl, pe, pp)

            loss = F.mse_loss(pred.float(), noise.float()) / accum_steps
            train_losses.append(loss.item() * accum_steps)
            loss.backward()

            # grad norms
            grad_norms = {
                f"grad_norm/{n}": p.grad.norm().item()
                for n, p in model.named_parameters() if p.grad is not None
            }

            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_batch_loss = np.mean(train_losses[-accum_steps:])
                log_d = {"train/loss": avg_batch_loss, "step": global_step}
                log_d.update(grad_norms)
                if use_wandb:
                    wandb.log(log_d, step=global_step)
                pbar.set_postfix(loss=avg_batch_loss)

        avg_train = float(np.mean(train_losses))
        if use_wandb:
            wandb.log({"epoch/train_loss": avg_train, "epoch": epoch+1}, step=global_step)

        # — Validation —
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                person   = batch["person_image"].to(device, dtype=torch.float16)
                mask     = batch["mask"].to(device, dtype=torch.float16)
                cloth    = batch["cloth_image"].to(device, dtype=torch.float16)
                tryon    = batch["overlay_image"].to(device, dtype=torch.float16)
                depth    = batch["depth_map"].to(device, dtype=torch.float16)
                normal   = batch["normal_map"].to(device, dtype=torch.float16)
                prompts  = batch["caption"]

                pe, pp = encode_prompt(model, model.tokenizer1, model.text_encoder1,
                                       model.tokenizer2, model.text_encoder2,
                                       model.tokenizer3, model.text_encoder3,
                                       prompts, device, False)
                ctrl = prepare_control_input(person, mask, cloth, model.vae, False)
                tgt  = prepare_target_latents(tryon, depth, normal, model.vae, False)
                noised, noise, t = add_noise(tgt, False)
                pred = model(noised, t, ctrl, pe, pp)

                v_loss = F.mse_loss(pred.float(), noise.float())
                val_losses.append(v_loss.item())

        avg_val = float(np.mean(val_losses))
        logger.info(f"Epoch {epoch+1}: train {avg_train:.4f}, val {avg_val:.4f}")
        if use_wandb:
            wandb.log({"epoch/val_loss": avg_val}, step=global_step)
        model.train()

        save_checkpoint(model, optimizer, global_step, checkpoint_dir=args.checkpoint_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
