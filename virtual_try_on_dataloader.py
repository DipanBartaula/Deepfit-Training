# virtual_try_on_dataloader.py

import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

# ---------------------- CONFIGURATION ----------------------
IMAGE_SIZE = (512, 512)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Transforms for RGB images
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

# Transforms for grayscale images (depth, mask)
basic_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


class VirtualTryOnDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with subfolders: image, cloth, normal, depth, mask, caption
        """
        self.root_dir = root_dir
        self.person_dir = os.path.join(root_dir, "image")
        self.cloth_dir  = os.path.join(root_dir, "cloth")
        self.normal_dir = os.path.join(root_dir, "normal")
        self.depth_dir  = os.path.join(root_dir, "depth")
        self.mask_dir   = os.path.join(root_dir, "mask")
        self.caption_dir= os.path.join(root_dir, "caption")

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
        person_path  = os.path.join(self.person_dir, img_file)
        cloth_path   = os.path.join(self.cloth_dir, img_file)
        normal_path  = os.path.join(self.normal_dir, f"{base_name}.jpg")
        depth_path   = os.path.join(self.depth_dir, f"{base_name}.jpg")
        mask_path    = os.path.join(self.mask_dir, f"{base_name}.png")
        caption_path = os.path.join(self.caption_dir, f"{base_name}.txt")

        # Load images
        person_pil = Image.open(person_path).convert("RGB")
        cloth_pil  = Image.open(cloth_path).convert("RGB")
        normal_pil = Image.open(normal_path).convert("RGB")
        depth_pil  = Image.open(depth_path).convert("L")
        mask_pil   = Image.open(mask_path).convert("L")

        # Create overlay
        person_np = np.array(person_pil)
        mask_np   = np.array(mask_pil)
        blurred   = cv2.GaussianBlur(mask_np, (21,21), sigmaX=10)
        alpha     = (blurred.astype(np.float32)/255.0)[...,None]
        grey      = np.full_like(person_np, 128, dtype=np.uint8)
        overlay_np= (person_np*(1-alpha) + grey*alpha).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        # Apply transforms
        person_image  = transform(person_pil)
        cloth_image   = transform(cloth_pil)
        normal_map    = transform(normal_pil)
        depth_map     = basic_transform(depth_pil)
        mask_tensor   = basic_transform(mask_pil).squeeze(0)
        overlay_image = transform(overlay_pil)

        # Load caption
        caption = ""
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        except FileNotFoundError:
            pass

        return {
            'person_image':  person_image,
            'cloth_image':   cloth_image,
            'normal_map':    normal_map,
            'depth_map':     depth_map,
            'mask':          mask_tensor,
            'overlay_image': overlay_image,
            'caption':       caption,
            'filename':      base_name
        }


def virtual_try_on_collate_fn(batch):
    """
    Collate that blurs masks, optionally includes captions, and stacks all tensors.
    """
    include_caption = (random.random() < 0.2)

    person_images, cloth_images, normal_maps = [], [], []
    depth_maps, masks, overlay_images         = [], [], []
    captions, filenames                       = [], []

    for item in batch:
        # Blur mask
        mask_np = item['mask'].cpu().numpy().astype(np.uint8)
        blurred = cv2.GaussianBlur(mask_np, (39,39), 0).astype(np.float32) / 255.0
        mask_t  = torch.from_numpy(blurred)

        person_images.append(   item['person_image'])
        cloth_images.append(    item['cloth_image'])
        normal_maps.append(     item['normal_map'])
        depth_maps.append(      item['depth_map'])
        masks.append(           mask_t)
        overlay_images.append(  item['overlay_image'])
        captions.append(        item['caption'] if include_caption else "")
        filenames.append(       item['filename'])

    return {
        'person_image':  torch.stack(person_images),
        'cloth_image':   torch.stack(cloth_images),
        'normal_map':    torch.stack(normal_maps),
        'depth_map':     torch.stack(depth_maps),
        'mask':          torch.stack(masks),
        'overlay_image': torch.stack(overlay_images),
        'caption':       captions,
        'filename':      filenames
    }


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
