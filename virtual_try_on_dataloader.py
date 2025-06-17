import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torch.multiprocessing import freeze_support
from torch.utils.data import random_split
from torchvision.transforms import functional as F


# ---------------------- CONFIGURATION ----------------------
IMAGE_SIZE = (512, 512)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

basic_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

class VirtualTryOnDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.person_dir = os.path.join(root_dir, "image")
        self.cloth_dir = os.path.join(root_dir, "cloth")
        self.normal_dir = os.path.join(root_dir, "normal")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.mask_dir = os.path.join(root_dir, "mask")
        self.caption_dir = os.path.join(root_dir, "caption")
        self.embed_dir = os.path.join(root_dir, "caption_embeds")

        self.image_files = [
            f for f in os.listdir(self.person_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        base_name, _ = os.path.splitext(img_file)

        # File paths
        person_path = os.path.join(self.person_dir, img_file)
        cloth_base = base_name.replace("_0", "_1")
        cloth_filename = cloth_base + ".jpg"
        cloth_path = os.path.join(self.cloth_dir, cloth_filename)
        # cloth_path = os.path.join(self.cloth_dir, img_file)
        normal_path = os.path.join(self.normal_dir, f"{base_name}.jpg")
        depth_path = os.path.join(self.depth_dir, f"{base_name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
        caption_path = os.path.join(self.caption_dir, f"{base_name}.txt")
        embed_path = os.path.join(self.embed_dir, f"{base_name}.npz")

        # Load images
        person_image_pil = Image.open(person_path).convert("RGB")
        cloth_image = Image.open(cloth_path).convert("RGB")
        normal_map = Image.open(normal_path).convert("RGB")
        depth_map = Image.open(depth_path).convert("L")
        mask_pil = Image.open(mask_path).convert("L")

        # Overlay image
        person_np = np.array(person_image_pil)
        mask_np = np.array(mask_pil)
        blurred_mask = cv2.GaussianBlur(mask_np, (21, 21), sigmaX=10)
        alpha_mask = np.expand_dims(blurred_mask.astype(np.float32) / 255.0, axis=2)
        grey_overlay = np.full_like(person_np, 128, dtype=np.uint8)
        overlay_np = (person_np * (1 - alpha_mask) + grey_overlay * alpha_mask).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        # Transform everything
        person_image = transform(person_image_pil)
        cloth_image = transform(cloth_image)
        normal_map = transform(normal_map)
        depth_map = basic_transform(depth_map)
        mask = basic_transform(mask_pil).squeeze(0)
        overlay_image = transform(overlay_pil)

        # Load caption
        caption = ""
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        except FileNotFoundError:
            pass

        # Load precomputed embedding
        try:
            data = np.load(embed_path)
            pe = torch.tensor(data['pe']).float()     # [77, D]
            pp = torch.tensor(data['pp']).float()     # [D]
        except FileNotFoundError:
            pe = torch.zeros(77, 1024)
            pp = torch.zeros(1024)

        return {
            'person_image': person_image,
            'cloth_image': cloth_image,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'mask': mask,
            'overlay_image': overlay_image,
            'caption': caption,
            'filename': base_name,
            'prompt_embeds': pe,
            'pooled_prompt': pp
        }

def virtual_try_on_collate_fn(batch):
    include_caption = (random.random() < 0.2)

    batch_dict = {
        'person_image':  torch.stack([b['person_image'] for b in batch]),
        'cloth_image':   torch.stack([b['cloth_image'] for b in batch]),
        'normal_map':    torch.stack([b['normal_map'] for b in batch]),
        'depth_map':     torch.stack([b['depth_map'] for b in batch]),
        'mask':          torch.stack([
            torch.from_numpy(cv2.GaussianBlur(b['mask'].cpu().numpy().astype(np.uint8), (39, 39), 0).astype(np.float32) / 255.0)
            for b in batch
        ]),
        'overlay_image': torch.stack([b['overlay_image'] for b in batch]),
        'caption':       [b['caption'] if include_caption else "" for b in batch],
        'filename':      [b['filename'] for b in batch],
        'pe':            torch.stack([b['prompt_embeds'] for b in batch]),  # [B, 77, D]
        'pp':            torch.stack([b['pooled_prompt'] for b in batch])   # [B, D]
    }
    return batch_dict

# ----------- Example usage -------------



    dataset = CustomDataset(root_dir="D:\\Pul - DeepFit\\Test\\dresses")
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # One batch test
    for batch in dataloader:
        print("Person:",   batch["person_image"].shape)
        print("Cloth:",    batch["cloth_image"].shape)
        print("Normal:",   batch["normal_map"].shape)
        print("Depth:",    batch["depth_map"].shape)
        print("Mask:",     batch["mask"].shape)
        print("Overlay:",  batch["overlay_image"].shape)
        print("PE:",       batch["prompt_embeds"])   # [B, 77, D]
        print("PP:",       batch["pooled_prompt"])   # [B, D]
        print("Captions:", batch["caption"])
        break




 

def get_train_val_loaders(
    root_dir: str,
    batch_size: int,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
):
    """
    Returns (train_loader, val_loader) for VirtualTryOnDataset.
    """
    ds = VirtualTryOnDataset(root_dir)
    n = len(ds)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=virtual_try_on_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=virtual_try_on_collate_fn,
        num_workers=max(1, num_workers//2),
        pin_memory=True
    )
    return train_loader, val_loader


def get_dataloader(
    root_dir: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
):
    """
    Returns a single DataLoader over the full VirtualTryOnDataset (no validation split).
    """
    ds = VirtualTryOnDataset(root_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=virtual_try_on_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
