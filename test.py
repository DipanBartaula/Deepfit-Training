# Quick sanity check (in a Python REPL or a small script):
import torch
from model import DeepFit
from utils import encode_prompt, prepare_control_input

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepFit(device=device, debug=True).to(device)
model.eval()

# Dummy batch:
B = 2
C, H, W = 3, 256, 256
# Random images:
person = torch.randn(B, C, H, W, device=device, dtype=torch.float16)
mask = (torch.rand(B, 1, H, W, device=device) > 0.5).to(torch.float16)
clothing = torch.randn(B, C, H, W, device=device, dtype=torch.float16)
prompts = ["A test prompt"] * B

# Encode prompt
prompt_embeds, pooled = encode_prompt(
    model=model,
    tokenizer1=model.tokenizer1,
    text_encoder1=model.text_encoder1,
    tokenizer2=model.tokenizer2,
    text_encoder2=model.text_encoder2,
    tokenizer3=model.tokenizer3,
    text_encoder3=model.text_encoder3,
    prompts=prompts,
    device=device,
    debug=True
)

# Prepare control_input
control_input = prepare_control_input(person, mask, clothing, vae=model.vae, debug=True)
# Encode tryon_gt dummy:
tryon_gt = torch.randn(B, C, H, W, device=device, dtype=torch.float16)
latent_dist = model.vae.encode(tryon_gt).latent_dist
img_latents = latent_dist.sample()
img_latents = (img_latents - model.vae.config.shift_factor) * model.vae.config.scaling_factor  # [B,16,H/8,W/8]
timesteps = torch.rand(B, device=device)
noise = torch.randn_like(img_latents)
noisy_img = img_latents + noise * timesteps[:, None, None, None]

# Forward
pred_noise = model(noisy_img, timesteps, control_input, prompt_embeds, pooled)
print("pred_noise shape:", pred_noise.shape)  # expect [B,16,H/8,W/8]
