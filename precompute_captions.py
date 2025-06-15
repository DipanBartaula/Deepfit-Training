#Datapipeline for precomputing caption embeddings for Stable Diffusion 3 Medium






import os
import torch.nn as nn
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, T5TokenizerFast, T5EncoderModel
from utils import encode_prompt

# adjust these paths & batch size as needed
ROOT = "D:\\PUL - DeepFit\\Dresscode"
BATCH_SIZE = 4
device = "cuda"

torch.cuda.empty_cache()

model = nn.Identity()

tokenizer1 = CLIPTokenizer.from_pretrained(
            "D:\\PUL - DeepFit\\Training\\stable-diffusion-3-medium-diffusers",
            subfolder="tokenizer"
        )

text_encoder1 = CLIPTextModelWithProjection.from_pretrained(
            "D:\\PUL - DeepFit\\Training\\stable-diffusion-3-medium-diffusers",
            subfolder="text_encoder",
            torch_dtype=torch.float16
        ).to(device)

tokenizer2 = CLIPTokenizer.from_pretrained(
            "D:\\PUL - DeepFit\\Training\\stable-diffusion-3-medium-diffusers",
            subfolder="tokenizer_2"
        )

text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
            "D:\\PUL - DeepFit\\Training\\stable-diffusion-3-medium-diffusers",
            subfolder="text_encoder_2",
            torch_dtype=torch.float16
        ).to(device)

tokenizer3 = T5TokenizerFast.from_pretrained(
            "D:\\PUL - DeepFit\\Training\\stable-diffusion-3-medium-diffusers",
            subfolder="tokenizer_3"
        )

text_encoder3 = T5EncoderModel.from_pretrained(
            "D:\\PUL - DeepFit\\Training\\stable-diffusion-3-medium-diffusers",
            subfolder="text_encoder_3",
            torch_dtype=torch.float16
        ).to(device)

for part in os.listdir(ROOT):
    part_dir = os.path.join(ROOT, part)
    if not os.path.isdir(part_dir):
        continue

    cap_dir = os.path.join(part_dir, "caption")
    if not os.path.isdir(cap_dir):
        continue

    # collect all base names
    files = sorted(f for f in os.listdir(cap_dir) if f.endswith(".txt"))
    names = [os.path.splitext(f)[0] for f in files]

    # read all captions
    captions = []
    for name in names:
        with open(os.path.join(cap_dir, name + ".txt"), "r", encoding="utf-8") as f:
            captions.append(f.read().strip())

    # batch‑encode
    pes, pps = [], []
    for i in range(0, len(captions), BATCH_SIZE):
        batch = captions[i : i + BATCH_SIZE]
        pe, pp = encode_prompt(
            model,
            tokenizer1, text_encoder1,
            tokenizer2, text_encoder2,
            tokenizer3, text_encoder3,
            batch,
            device,
            debug=False,
        )
        pe = pe.detach().cpu().half().numpy()
        pp = pp.detach().cpu().half().numpy()
        pes.append(pe)
        pps.append(pp)

    # concatenate
    all_pe = np.concatenate(pes, axis=0)
    all_pp = np.concatenate(pps, axis=0)

    # make caption_embeds folder
    embed_dir = os.path.join(part_dir, "caption_embeds")
    os.makedirs(embed_dir, exist_ok=True)

    # save .npz
    out_path = os.path.join(embed_dir, "precomputed_prompts.npz")
    np.savez_compressed(out_path, pe=all_pe, pp=all_pp, names=np.array(names))

    

    print(f"[{part}] saved {all_pe.shape} → {out_path}")