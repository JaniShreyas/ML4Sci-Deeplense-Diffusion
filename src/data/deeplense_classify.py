from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, Dataloader
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
    pass

deeplense_classify_MEAN, deeplense_classify_STD = get_stats()

def denormalize(x: torch.Tensor):
    pass

def default_transform(cfg: DataConfig | None = None):
    pass

def get_datasets(cfg: DataConfig) -> tuple:
    pass

def get_dataloaders(cfg: DataConfig) -> tuple:
    train_ds, test_ds = get_datasets(cfg)
    return make_dataloaders(train_ds, test_ds, cfg)