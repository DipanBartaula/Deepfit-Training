# utils.py
"""
Utility functions: loading VAE, schedulers, checkpoint manager, checkpoint discovery.
"""
import os
import glob
import torch
from diffusers import AutoencoderKL, DDPMScheduler

def load_vae(device: str = "cuda"):
    """
    Load pretrained VAE from Stable Diffusion v2.
    """
    print("[utils] Loading VAE from Stable Diffusion v2...")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    print("[utils] VAE loaded.")
    return vae

class CheckpointManager:
    """
    Saves/loads model + optimizer + scaler states.
    Also can discover latest checkpoint in a directory.
    """
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def save(self, model, optimizer, scaler, epoch: int = None, step: int = None):
        """
        Save checkpoint. If step is provided, name includes step; else if epoch provided, name includes epoch.
        """
        if step is not None:
            fname = f"checkpoint_step_{step}.pt"
        elif epoch is not None:
            fname = f"checkpoint_epoch_{epoch}.pt"
        else:
            raise ValueError("Either epoch or step must be provided for checkpoint naming.")
        path = os.path.join(self.output_dir, fname)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
        }
        torch.save(ckpt, path)
        print(f"[utils] Checkpoint saved: {path}")

    def load(self, path: str, model, optimizer=None, scaler=None):
        """
        Load checkpoint from given path into model, optimizer, scaler.
        Returns loaded epoch and step.
        """
        print(f"[utils] Loading checkpoint from: {path}")
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if optimizer and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        epoch = ckpt.get("epoch", None)
        step = ckpt.get("step", None)
        print(f"[utils] Loaded checkpoint epoch: {epoch}, step: {step}")
        return epoch, step

    def get_latest_checkpoint(self):
        """
        Discover latest checkpoint in output_dir by highest step or epoch.
        Prefers step checkpoints over epoch if both present; compares numeric suffix.
        Returns path or None.
        """
        pattern_step = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        pattern_epoch = os.path.join(self.output_dir, "checkpoint_epoch_*.pt")
        files = glob.glob(pattern_step)
        latest = None
        mode = None
        if files:
            steps = []
            for f in files:
                base = os.path.basename(f)
                try:
                    n = int(base.split("_")[2].split(".")[0])
                    steps.append((n, f))
                except:
                    continue
            if steps:
                steps.sort(key=lambda x: x[0])
                latest = steps[-1][1]
                mode = 'step'
        if latest is None:
            files_e = glob.glob(pattern_epoch)
            if files_e:
                epochs = []
                for f in files_e:
                    base = os.path.basename(f)
                    try:
                        n = int(base.split("_")[2].split(".")[0])
                        epochs.append((n, f))
                    except:
                        continue
                if epochs:
                    epochs.sort(key=lambda x: x[0])
                    latest = epochs[-1][1]
                    mode = 'epoch'
        if latest:
            print(f"[utils] Latest checkpoint ({mode}) found: {latest}")
        else:
            print("[utils] No checkpoint found.")
        return latest

def get_noise_scheduler(
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    beta_schedule: str = "scaled_linear",
    num_train_timesteps: int = 1000
):
    """
    Initialize and return a DDPMScheduler for noise addition/removal.
    """
    print("[utils] Initializing noise scheduler...")
    scheduler = DDPMScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        num_train_timesteps=num_train_timesteps
    )
    print("[utils] Noise scheduler ready.")
    return scheduler
