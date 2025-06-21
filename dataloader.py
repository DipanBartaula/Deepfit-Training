# dataloader.py
"""
Dummy data loader for Virtual Try-On. Replace with real data pipeline.
Outputs per sample:
- overlay_image: (3, H, W)
- cloth_image:   (3, H, W)
- mask:          (1, H, W)
- surface_normal:(3, H, W)
- depth_map:     (1, H, W)
- text_embeddings: (seq_len, embed_dim)
"""
import torch
from torch.utils.data import Dataset, DataLoader

class DummyVirtualTryOnDataset(Dataset):
    def __init__(self, length=1000, image_size=(256,256), seq_len=77, embed_dim=768, device="cpu"):
        self.length = length
        self.H, self.W = image_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        overlay_image = torch.randn(3, self.H, self.W, device=self.device)
        cloth_image   = torch.randn(3, self.H, self.W, device=self.device)
        mask          = torch.randint(0,2,(1,self.H,self.W),dtype=torch.float32,device=self.device)
        surface_normal= torch.randn(3, self.H, self.W, device=self.device)
        depth_map     = torch.rand(1, self.H, self.W, device=self.device)
        text_embeddings = torch.randn(self.seq_len, self.embed_dim, device=self.device)
        return {
            "overlay_image": overlay_image,
            "cloth_image": cloth_image,
            "mask": mask,
            "surface_normal": surface_normal,
            "depth_map": depth_map,
            "text_embeddings": text_embeddings,
        }

def get_dataloader(batch_size=4, num_workers=4, device="cuda"):
    """
    Returns DataLoader wrapping DummyVirtualTryOnDataset.
    """
    print(f"[dataloader] Creating DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    dataset = DummyVirtualTryOnDataset(device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("[dataloader] DataLoader ready.")
    return loader
