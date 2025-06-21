# test.py
"""
Test DeepFit forward pass and trainable parameter check.
"""
import torch
from model import DeepFit
from utils import load_vae
from dataloader import DummyVirtualTryOnDataset

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_module = DeepFit(train_self_attention_only=True).to(device)
    unet = model_module.unet

    trainable = [n for n,p in unet.named_parameters() if p.requires_grad]
    frozen    = [n for n,p in unet.named_parameters() if not p.requires_grad]
    print(f"[test] Trainable ({len(trainable)}): {trainable[:5]}…")
    print(f"[test] Frozen    ({len(frozen)})")

    ds = DummyVirtualTryOnDataset(length=1, device=device)
    s = ds[0]
    ov,cl,sn,dp,mk,te = [s[k].unsqueeze(0) for k in ["overlay_image","cloth_image","surface_normal","depth_map","mask","text_embeddings"]]

    vae = load_vae(device)
    from train import prepare_concat_latents
    inp, _ = prepare_concat_latents(vae, ov, cl, sn, dp, mk)
    t = torch.randint(0,1000,(1,),device=device).long()
    out = unet(inp, t, encoder_hidden_states=te)
    print("[test] Output tensor shape:", out.shape)

if __name__ == "__main__":
    test()
