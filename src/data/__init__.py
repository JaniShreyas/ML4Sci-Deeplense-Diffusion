from torch import Tensor

from .mnist import get_dataloaders as get_mnist_dataloaders, get_stats as get_mnist_stats, get_datasets as get_mnist_datasets, denormalize as denormalize_mnist
from .cifar10 import get_dataloaders as get_cifar10_dataloaders, get_stats as get_cifar10_stats, get_datasets as get_cifar10_datasets, denormalize as denormalize_cifar10
from .deeplense_classify import get_dataloaders as get_deeplense_classify_dataloaders, get_stats as get_deeplense_classify_stats, get_datasets as get_deeplense_classify_datasets, denormalize as denormalize_deeplense_classify
from .deeplense_diffusion import get_dataloaders as get_deeplense_diffusion_dataloaders, get_stats as get_deeplense_diffusion_stats, get_datasets as get_deeplense_diffusion_datasets, denormalize as denormalize_deeplense_diffusion

from .config import DataConfig
from torch.utils.data import DataLoader, Dataset

# The dataset registry
DATASET_REGISTRY = {
    "mnist": (get_mnist_dataloaders, get_mnist_stats, get_mnist_datasets, denormalize_mnist),
    "cifar10": (get_cifar10_dataloaders, get_cifar10_stats, get_cifar10_datasets, denormalize_cifar10),
    "deeplense_classify": (get_deeplense_classify_dataloaders, get_deeplense_classify_stats, get_deeplense_classify_datasets, denormalize_deeplense_classify),
    "deeplense_diffusion": (get_deeplense_diffusion_dataloaders, get_deeplense_diffusion_stats, get_deeplense_diffusion_datasets, denormalize_deeplense_diffusion),
}

def verify_dataset(cfg_name: str) -> None:
    if cfg_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {cfg_name} is not supported. Available datasets are: {list(DATASET_REGISTRY.keys())}")

def get_dataloaders(cfg: DataConfig) -> DataLoader:
    print(f"Loading dataset: {cfg.name}")

    verify_dataset(cfg.name)
    
    dataloader_fn = DATASET_REGISTRY[cfg.name][0]

    return dataloader_fn(cfg)


def get_stats(cfg: DataConfig) -> tuple:
    verify_dataset(cfg.name)
    return DATASET_REGISTRY[cfg.name][1]()


def get_datasets(cfg: DataConfig) -> Dataset:
    verify_dataset(cfg.name)
    print(f"Loading dataset: {cfg.name}")

    dataset_fn = DATASET_REGISTRY[cfg.name][2]
    
    return dataset_fn(cfg)

def denormalize(cfg: DataConfig, x: Tensor) -> Tensor:
    verify_dataset(cfg.name)
    return DATASET_REGISTRY[cfg.name][3](x)