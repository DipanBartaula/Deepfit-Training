# DeepFit Virtual Try-On Inpainting Project

This project trains a DeepFit for virtual try-on tasks. It includes gradient checkpointing, gradient accumulation, mixed precision, checkpointing every 100 iterations, resume support, and integration with Hugging Face and Accelerate.

## File Structure

- `model.py`: Defines `DeepFit` class loading pretrained inpainting UNet, freezing non-attention parameters.
- `utils.py`: Utilities: loading VAE, noise scheduler, checkpoint manager with discovery.
- `dataloader.py`: Dummy data loader; replace with real dataset.
- `train.py`: Training script using Accelerate for mixed precision and multi-GPU/distributed, gradient checkpointing, gradient accumulation, debug prints, checkpointing.
- `test.py`: Test forward pass and trainable parameter check.
- `inference.py`: Single-step inference demo; adapt for full reverse diffusion.
- `huggingface_login.py`: Utility to login to Hugging Face Hub using token from `.env`.
- `run_train.bat`: Windows script to launch training with gradient checkpointing enabled by default.
- `requirements.txt`: Python dependencies.
- `.env`: Define `HUGGINGFACE_TOKEN`.
- `README.md`: This file.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
