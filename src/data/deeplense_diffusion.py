import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import numpy as np
from pathlib import Path

from .config import DataConfig
from .utils import make_dataloaders


class DeepLenseDiffusionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform     
        self.filepaths = list(self.root_dir.glob("*.npy"))
        
        if len(self.filepaths) == 0:
            print(f"Warning: No .npy files found in {self.root_dir}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        image = torch.tensor(np.load(filepath), dtype=torch.float32)

        # Ensure channel dimension
        if image.ndim == 2:
            image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        # Return a dummy label (0) just in case. Though the trainer does support only images being returned
        return image, 0

def get_stats():
    return (0, ), (1, )  

deeplense_diffusion_MEAN, deeplense_diffusion_STD = get_stats()

def denormalize(x: torch.Tensor):
    mean = torch.tensor(deeplense_diffusion_MEAN, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.tensor(deeplense_diffusion_STD, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    return x

def default_transform(cfg: DataConfig | None = None):
    if cfg is None:
        cfg = DataConfig()
    if cfg.transform is not None:
        return cfg.transform
        
    t_list = []
    
    if cfg.image_size:
        size = cfg.image_size if isinstance(cfg.image_size, tuple) else (cfg.image_size, cfg.image_size)
        t_list.append(transforms.Resize(size))
        
    t_list.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])

    return transforms.Compose(t_list)

def get_datasets(cfg: DataConfig) -> tuple:
    transform = cfg.transform or default_transform(cfg)
    full_ds = DeepLenseDiffusionDataset(root_dir=Path(cfg.data_root) / "samples", transform=transform)
    
    total_size = len(full_ds)
    val_size = max(1, int(0.1 * total_size)) 
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    
    return train_ds, val_ds

def get_dataloaders(cfg: DataConfig) -> tuple:
    train_ds, val_ds = get_datasets(cfg)
    return make_dataloaders(train_ds, val_ds, cfg)