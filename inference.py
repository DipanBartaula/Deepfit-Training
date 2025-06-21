# inference.py
"""
Single-step inference demo for DeepFit. Replace with full scheduler loop for real use.
"""
import torch
import torch.nn.functional as F
from model import DeepFit
from utils import load_vae, get_noise_scheduler
from dataloader import DummyVirtualTryOnDataset

def inference_step(model, vae, sched, ov, cl, sn, dp, mk, te):
    """
    One-step demo: encode clean inputs, add max noise, denoise once, decode.
    For true inference, implement a full reverse diffusion loop.
    """
    from train import prepare_concat_latents
    inp_clean, ov_lat = prepare_concat_latents(vae, ov, cl, sn, dp, mk)
    t = torch.tensor([sched.num_train_timesteps - 1], device=ov.device).long()
    noise = torch.randn_like(ov_lat)
    noisy = sched.add_noise(ov_lat, noise, t)

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

    pred = model(inp_noisy, t, encoder_hidden_states=te)
    overlay_w = ov_lat.shape[3]
    pn = pred[:, :4, :, :overlay_w]
    clean = noisy - pn
    with torch.no_grad():
        out = vae.decode(clean / vae.config.scaling_factor).sample
    return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(device)
    model = DeepFit(train_self_attention_only=True).to(device).unet
    sched = get_noise_scheduler()
    ds = DummyVirtualTryOnDataset(length=1, device=device)
    s = ds[0]
    ov,cl,sn,dp,mk,te = [s[k].unsqueeze(0) for k in ["overlay_image","cloth_image","surface_normal","depth_map","mask","text_embeddings"]]
    out = inference_step(model, vae, sched, ov, cl, sn, dp, mk, te)
    print("[inference] Output shape:", out.shape)
