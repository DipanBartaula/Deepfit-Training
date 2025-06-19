import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# ---------------------- CONFIGURATION ----------------------
IMAGE_SIZE = (1024,768)

NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Path to a default .npz that contains embeddings for the empty prompt ("")
# Must contain arrays `prompt_embed` and `pooled_prompt`
DEFAULT_EMBED_PATH = "/path/to/default_empty_prompt.npz"

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
    """
    Multi‑category dataset: loads samples from all given subfolders
    in a train directory (e.g., 'dresses', 'upper_body', etc.).
    """
    def __init__(self, root_dir: str, categories: list):
        self.root_dir = root_dir
        self.categories = categories
        self.index = []  # list of (category, base_name)
        for cat in categories:
            img_dir = os.path.join(root_dir, cat, "image")
            if not os.path.isdir(img_dir):
                continue
            for f in os.listdir(img_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    base_name, _ = os.path.splitext(f)
                    self.index.append((cat, base_name))
        print(f"[DEBUG] VirtualTryOnDataset loaded {len(self.index)} samples from {root_dir} over categories {categories}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cat, base_name = self.index[idx]
        base_dir = os.path.join(self.root_dir, cat)

        # Build file paths
        person_path  = os.path.join(base_dir, "image", f"{base_name}.jpg")
        cloth_path   = os.path.join(base_dir, "cloth", f"{base_name}.jpg")
        normal_path  = os.path.join(base_dir, "normal", f"{base_name}.jpg")
        depth_path   = os.path.join(base_dir, "depth", f"{base_name}.jpg")
        mask_path    = os.path.join(base_dir, "mask", f"{base_name}.png")
        caption_path = os.path.join(base_dir, "caption", f"{base_name}.txt")
        embed_path   = os.path.join(base_dir, "caption_embeds", f"{base_name}.npz")

        # Load and process images
        person_image_pil = Image.open(person_path).convert("RGB")
        cloth_image      = Image.open(cloth_path).convert("RGB")
        normal_map       = Image.open(normal_path).convert("RGB")
        depth_map        = Image.open(depth_path).convert("L")
        mask_pil         = Image.open(mask_path).convert("L")

        # Gray overlay
        person_np = np.array(person_image_pil)
        mask_np   = np.array(mask_pil)
        blurred   = cv2.GaussianBlur(mask_np, (21, 21), sigmaX=10)
        alpha     = np.expand_dims(blurred.astype(np.float32) / 255.0, 2)
        grey      = np.full_like(person_np, 128, np.uint8)
        overlay_np = (person_np * (1 - alpha) + grey * alpha).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        # Transforms
        person_image  = person_image_pil.ToTensor()
        person_image=person_image.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        cloth_image   = transform(cloth_image)
        normal_map  = normal_map.ToTensor()
        normal_map=normal_map.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        depth_map  = depth_map.ToTensor()
        depth_map=depth_map.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        mask_tensor   = basic_transform(mask_pil)
        overlay_image = transform(overlay_pil)

        # Load caption text
        caption = ""
        if os.path.isfile(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()

        # Load embeddings (possibly with leading batch dim)
        if os.path.isfile(embed_path):
            data = np.load(embed_path)
            pe = torch.tensor(data.get("prompt_embed", data.get("pe"))).float()
            pp = torch.tensor(data.get("pooled_prompt", data.get("pp"))).float()
        else:
            data = np.load(DEFAULT_EMBED_PATH)
            pe = torch.tensor(data["prompt_embed"]).float()
            pp = torch.tensor(data["pooled_prompt"]).float()

        # Ensure shapes: drop any leading singleton batch dimension
        if pe.ndim == 3 and pe.size(0) == 1:
            pe = pe.squeeze(0)
        if pp.ndim == 2 and pp.size(0) == 1:
            pp = pp.squeeze(0)

        return {
            "person_image":  person_image,
            "cloth_image":   cloth_image,
            "normal_map":    normal_map,
            "depth_map":     depth_map,
            "mask":          mask_tensor,
            "overlay_image": overlay_image,
            "caption":       caption,
            "filename":      base_name,
            "category":      cat,
            "prompt_embeds": pe,
            "pooled_prompt": pp
        }


class ValidationDataset(Dataset):
    """
    Validation dataset: flat directory with subfolders 'image', 'cloth', etc.
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "image")
        if not os.path.isdir(self.image_dir):
            contents = os.listdir(root_dir)
            raise FileNotFoundError(
                f"Validation root '{root_dir}' lacks an 'image' subfolder.\n"
                f"Found instead: {contents}"
            )
        self.files = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"[DEBUG] ValidationDataset loaded {len(self.files)} samples from {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        base_name, _ = os.path.splitext(img_file)

        # Paths
        person_path  = os.path.join(self.root_dir, "image", base_name + ".jpg")
        cloth_path   = os.path.join(self.root_dir, "cloth", base_name + ".jpg")
        normal_path  = os.path.join(self.root_dir, "normal", base_name + ".jpg")
        depth_path   = os.path.join(self.root_dir, "depth", base_name + ".jpg")
        mask_path    = os.path.join(self.root_dir, "mask", base_name + ".png")
        caption_path = os.path.join(self.root_dir, "caption", base_name + ".txt")
        embed_path   = os.path.join(self.root_dir, "caption_embeds", base_name + ".npz")

        # Load and process images
        person_image_pil = Image.open(person_path).convert("RGB")
        cloth_image      = Image.open(cloth_path).convert("RGB")
        normal_map       = Image.open(normal_path).convert("RGB")
        depth_map        = Image.open(depth_path).convert("L")
        mask_pil         = Image.open(mask_path).convert("L")

        person_np = np.array(person_image_pil)
        mask_np   = np.array(mask_pil)
        blurred   = cv2.GaussianBlur(mask_np, (21, 21), sigmaX=10)
        alpha     = np.expand_dims(blurred.astype(np.float32) / 255.0, 2)
        grey      = np.full_like(person_np, 128, np.uint8)
        overlay_np = (person_np * (1 - alpha) + grey * alpha).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        person_image  = person_image_pil.ToTensor()
        person_image=person_image.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        cloth_image   = transform(cloth_image)
        normal_map  = normal_map.ToTensor()
        normal_map=normal_map.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        depth_map  = depth_map.ToTensor()
        depth_map=depth_map.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        mask_tensor   = basic_transform(mask_pil)
        overlay_image = transform(overlay_pil)

        caption = ""
        if os.path.isfile(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()

        if os.path.isfile(embed_path):
            data = np.load(embed_path)
            pe = torch.tensor(data.get("prompt_embed", data.get("pe"))).float()
            pp = torch.tensor(data.get("pooled_prompt", data.get("pp"))).float()
        else:
            data = np.load(DEFAULT_EMBED_PATH)
            pe = torch.tensor(data["prompt_embed"]).float()
            pp = torch.tensor(data["pooled_prompt"]).float()

        if pe.ndim == 3 and pe.size(0) == 1:
            pe = pe.squeeze(0)
        if pp.ndim == 2 and pp.size(0) == 1:
            pp = pp.squeeze(0)

        return {
            "person_image":  person_image,
            "cloth_image":   cloth_image,
            "normal_map":    normal_map,
            "depth_map":     depth_map,
            "mask":          mask_tensor,
            "overlay_image": overlay_image,
            "caption":       caption,
            "filename":      base_name,
            "prompt_embeds": pe,
            "pooled_prompt": pp
        }


def virtual_try_on_collate_fn(batch):
    """Randomly drop captions (~20%) and zero out embeddings when captions are dropped."""
    keep_flags = [random.random() < 0.2 for _ in batch]

    # Stack image tensors
    stacked = {k: torch.stack([b[k] for b in batch])
               for k in ["person_image", "cloth_image", "normal_map", "depth_map", "overlay_image"]}

    masks = torch.stack([b['mask'] for b in batch])           

    # Process masks
    # masks = torch.stack([
    #     torch.from_numpy(
    #         cv2.GaussianBlur(b["mask"].cpu().numpy().astype(np.uint8), (39, 39), 0)
    #         .astype(np.float32) / 255.0
    #     )
    #     for b in batch
    # ])

    captions, pe_list, pp_list, filenames = [], [], [], []
    for b, keep in zip(batch, keep_flags):
        if keep and b["caption"]:
            captions.append(b["caption"])
            pe_list.append(b["prompt_embeds"])
            pp_list.append(b["pooled_prompt"])
        else:
            captions.append("")
            pe_list.append(torch.zeros_like(b["prompt_embeds"]))
            pp_list.append(torch.zeros_like(b["pooled_prompt"]))
        filenames.append(b["filename"])

    return {
        **stacked,
        "mask":          masks,
        "caption":       captions,
        "filename":      filenames,
        "prompt_embeds": torch.stack(pe_list),
        "pooled_prompt": torch.stack(pp_list),
    }


def get_dataloader(
    root_dir: str,
    categories: list,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    ds = VirtualTryOnDataset(root_dir, categories)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=virtual_try_on_collate_fn,
        num_workers=num_workers, pin_memory=True
    )


def get_validation_loader(
    val_root: str,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 2
) -> DataLoader:
    ds = ValidationDataset(val_root)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=virtual_try_on_collate_fn,
        num_workers=num_workers, pin_memory=True
    )


def get_train_val_loaders(
    train_root: str,
    val_root: str,
    categories: list,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 4
) -> (DataLoader, DataLoader):
    train_loader = get_dataloader(
        root_dir=train_root,
        categories=categories,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    val_loader = get_validation_loader(
        val_root=val_root,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2)
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # --------------------------
    # Quick test of both loaders
    # --------------------------
    cats = ["dresses", "upper_body", "lower_body", "upper_body1"]
    train_root = r"D:\PUL - DeepFit\Dresscode"
    val_root   = r"D:\PUL - DeepFit\Test"
    bs = 4

    print(">>> Testing VirtualTryOnDataset + DataLoader")
    train_loader, val_loader = get_train_val_loaders(
        train_root=train_root,
        val_root=val_root,
        categories=cats,
        batch_size=bs,
        shuffle_train=False,
        num_workers=0
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    train_batch = next(iter(train_loader))
    print("One train batch sizes:")
    for k, v in train_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)} len {len(v)}")


            
