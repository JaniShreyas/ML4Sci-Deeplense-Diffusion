from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from .config import DataConfig
from .utils import make_dataloaders


class DeepLenseClassifyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform     

        self.classes = ["no", "sphere", "vort"]
        self.classes_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.filepaths = []
        self.labels = []

        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for file_path in cls_dir.glob("*.npy"):
                self.filepaths.append(file_path)
                self.labels.append(self.classes_to_idx[cls])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        image = torch.tensor(np.load(filepath), dtype=torch.float32)

        if image.ndim == 2:
            image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_stats():
    # return (0.061692753292123474, ), (0.11580801538924376, )
    return (0, ), (1, )  # Since the images are already normalized to [0, 1], we can use mean=0 and std=1 for effectively no normalization 

deeplense_classify_MEAN, deeplense_classify_STD = get_stats()

def denormalize(x: torch.Tensor):
    mean = torch.tensor(deeplense_classify_MEAN, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.tensor(deeplense_classify_STD, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    return x

def default_transform(cfg: DataConfig | None = None):
    # We are told that by default min-max normalization has been used to scale the images
    # What else can be used
    if cfg is None:
        cfg = DataConfig()
    if cfg.transform is not None:
        return cfg.transform
    t_list = []
    if cfg.image_size:
        size = cfg.image_size
        if isinstance(size, int):
            t_list.append(transforms.Resize(size))
        else:
            t_list.append(transforms.Resize(size))

    return transforms.Compose(t_list)

def get_datasets(cfg: DataConfig) -> tuple:
    transform = cfg.transform or default_transform(cfg)
    train_ds = DeepLenseClassifyDataset(root_dir=Path(cfg.data_root) / "train", transform=transform)
    val_ds = DeepLenseClassifyDataset(root_dir=Path(cfg.data_root) / "val", transform=transform)
    return train_ds, val_ds

def get_dataloaders(cfg: DataConfig) -> tuple:
    train_ds, val_ds = get_datasets(cfg)
    return make_dataloaders(train_ds, val_ds, cfg)