# train.py
"""
Training script for DeepFit Virtual Try-On inpainting with:
- Mixed precision (torch.cuda.amp)
- Gradient checkpointing always enabled
- Gradient accumulation for effective batch size
- Debug print statements for tracing
- Saving checkpoints every 100 iterations
- Resuming from latest checkpoint if available
- Uses accelerate library optionally for multi-GPU or distributed setups
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Optional: import accelerate
from accelerate import Accelerator

from model import DeepFit
from utils import load_vae, get_noise_scheduler, CheckpointManager
from dataloader import get_dataloader

def prepare_concat_latents(vae, overlay_image, cloth_image, surface_normal, depth_map, mask):
    """
    Prepare concatenated latent input for UNet:
    1) Encode overlay_image & cloth_image via VAE: (B,4,h',w') each → spatial concat width: (B,4,h',2w')
    2) Encode surface_normal (3ch) & depth_map (1ch→3ch): each via VAE → (B,4,h',w') → spatial concat: (B,4,h',2w')
    3) Channel-concat those two: (B,8,h',2w')
    4) Resize mask to (h',w'), create dark zeros, spatial concat width: (B,1,h',2w')
    5) Channel-concat: (B,9,h',2w') → return final input and overlay_latent (clean) for noise target.
    """
    overlay_latent = vae.encode(overlay_image).latent_dist.sample() * vae.config.scaling_factor
    cloth_latent   = vae.encode(cloth_image).latent_dist.sample() * vae.config.scaling_factor
    ov_cl = torch.cat([overlay_latent, cloth_latent], dim=3)

    depth_rgb = depth_map.repeat(1,3,1,1)
    normal_latent = vae.encode(surface_normal).latent_dist.sample() * vae.config.scaling_factor
    depth_latent  = vae.encode(depth_rgb).latent_dist.sample() * vae.config.scaling_factor
    nd = torch.cat([normal_latent, depth_latent], dim=3)

    comb = torch.cat([ov_cl, nd], dim=1)

    h, w2 = comb.shape[2], comb.shape[3] // 2
    mask_lat = torch.nn.functional.interpolate(mask, size=(h, w2), mode="nearest")
    dark = torch.zeros_like(mask_lat)
    md = torch.cat([mask_lat, dark], dim=3)

    final = torch.cat([comb, md], dim=1)
    return final, overlay_latent

def train(args):
    # Initialize Accelerator for mixed precision and multi-GPU/distributed if desired
    accelerator = Accelerator(mixed_precision='fp16' if torch.cuda.is_available() else None)
    device = accelerator.device
    print(f"[train] Using device from Accelerator: {device}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    vae = load_vae(device)

    # Initialize DeepFit UNet
    print("[train] Initializing DeepFit UNet...")
    model_module = DeepFit(pretrained_model_name=args.pretrained, train_self_attention_only=True)
    unet = model_module.unet

    # Enable gradient checkpointing always
    print("[train] Enabling gradient checkpointing on UNet...")
    unet.enable_gradient_checkpointing()

    # Setup optimizer on trainable parameters
    trainable = [p for p in unet.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable)
    print(f"[train] Number of trainable parameters: {total_trainable}")
    optimizer = AdamW(trainable, lr=args.lr)

    # Scheduler and scaler
    scheduler = get_noise_scheduler()
    scaler = GradScaler()
    ckpt_mgr = CheckpointManager(args.output_dir)

    # Attempt resume from latest checkpoint
    latest_ckpt = ckpt_mgr.get_latest_checkpoint()
    start_epoch = 1
    global_step = 0
    if latest_ckpt:
        try:
            epoch_loaded, step_loaded = ckpt_mgr.load(latest_ckpt, unet, optimizer, scaler)
            if epoch_loaded is not None:
                start_epoch = epoch_loaded
            if step_loaded is not None:
                global_step = step_loaded
            print(f"[train] Resuming from epoch {start_epoch}, global_step {global_step}")
        except Exception as e:
            print(f"[train] Warning: failed to load checkpoint {latest_ckpt}: {e}")

    # DataLoader
    loader = get_dataloader(args.batch_size, args.num_workers, device)

    # Gradient accumulation settings
    if args.effective_batch_size % args.batch_size != 0:
        raise ValueError(f"Effective batch size ({args.effective_batch_size}) must be divisible by micro batch size ({args.batch_size})")
    accum_steps = args.effective_batch_size // args.batch_size
    print(f"[train] Gradient accumulation: micro batch size={args.batch_size}, effective batch size={args.effective_batch_size}, accumulation steps={accum_steps}")

    # Prepare everything with accelerator
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    # Training loop
    for epoch in range(start_epoch, args.epochs+1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")
        unet.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            ov = batch["overlay_image"].to(device)
            cl = batch["cloth_image"].to(device)
            sn = batch["surface_normal"].to(device)
            dp = batch["depth_map"].to(device)
            mk = batch["mask"].to(device)
            te = batch["text_embeddings"].to(device)

            # Prepare inputs
            inp_clean, ov_lat = prepare_concat_latents(vae, ov, cl, sn, dp, mk)
            B = ov_lat.shape[0]
            print(f"[train] Batch {step}: overlay_latent shape={ov_lat.shape}, inp_clean shape={inp_clean.shape}")

            # Sample timesteps and add noise
            t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
            noise = torch.randn_like(ov_lat)
            noisy = scheduler.add_noise(ov_lat, noise, t)
            print(f"[train]   Timesteps shape={t.shape}, noisy latent shape={noisy.shape}")

            # Rebuild noisy input
            cl_lat = vae.encode(cl).latent_dist.sample() * vae.config.scaling_factor
            ov_ns = torch.cat([noisy, cl_lat], dim=3)
            sn_lat = vae.encode(sn).latent_dist.sample() * vae.config.scaling_factor
            dp_rgb = dp.repeat(1,3,1,1)
            dp_lat = vae.encode(dp_rgb).latent_dist.sample() * vae.config.scaling_factor
            nd_sp = torch.cat([sn_lat, dp_lat], dim=3)
            cmb = torch.cat([ov_ns, nd_sp], dim=1)
            h, w2 = cmb.shape[2], cmb.shape[3] // 2
            mk_lat = torch.nn.functional.interpolate(mk, size=(h, w2), mode="nearest")
            dk = torch.zeros_like(mk_lat)
            md_sp = torch.cat([mk_lat, dk], dim=3)
            inp_noisy = torch.cat([cmb, md_sp], dim=1)
            print(f"[train]   inp_noisy shape={inp_noisy.shape}")

            # Forward + loss
            with autocast():
                pred = unet(inp_noisy, t, encoder_hidden_states=te)
                pad = torch.zeros((B,4,ov_lat.shape[2],ov_lat.shape[3]), device=device)
                noise_pad = torch.cat([noise, pad], dim=3)
                target = torch.zeros_like(pred)
                target[:, :4] = noise_pad
                loss = F.mse_loss(pred, target) / accum_steps
            print(f"[train]   Loss/divided={loss.item():.6f}")

            accelerator.backward(loss)

            # Gradient accumulation and optimizer step
            if (step + 1) % accum_steps == 0:
                global_step += 1
                print(f"[train]   Performing optimizer.step() at global step {global_step}")
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            else:
                print(f"[train]   Accumulating gradients: step {step+1}/{accum_steps}")
                global_step += 1

            # Save checkpoint every 100 iterations
            if global_step % 100 == 0:
                print(f"[train]   Saving checkpoint at global step {global_step}")
                ckpt_mgr.save(unet, optimizer, scaler=None, epoch=epoch, step=global_step)

            pbar.set_postfix({"loss": loss.item(), "step": global_step})

        # End of epoch: leftover grads
        if len(loader) % accum_steps != 0:
            print("[train]   Final optimizer step for leftover gradients at epoch end")
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Save epoch checkpoint
        print(f"[train] Saving end-of-epoch checkpoint for epoch {epoch}")
        ckpt_mgr.save(unet, optimizer, scaler=None, epoch=epoch, step=global_step)
        print(f"[train] Epoch {epoch} checkpoint saved.")

    print("[train] Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train DeepFit Virtual Try-On Inpainting Model with Accelerate")
    parser.add_argument("--pretrained", type=str, default="stabilityai/stable-diffusion-2-inpainting",
                        help="Hugging Face model identifier for Stable Diffusion v2 inpainting UNet")
    parser.add_argument("--batch_size", type=int, required=True, help="Micro batch size per step")
    parser.add_argument("--effective_batch_size", type=int, required=True, help="Desired effective batch size for gradient accumulation")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping; set <=0 to disable")
    args = parser.parse_args()
    train(args)
