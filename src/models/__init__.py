from src.models.backbones.unet import UNet
from src.models.backbones.unet_attention import UNetWithAttention
from src.models.ddpm import DiffusionModel
from src.models.backbones.jit import JiT
from src.models.efficient_net_b2 import EfficientNetB2
from src.models.PINN import PhysicsInformedClassifier

import torch.nn as nn

# Builder functions

def create_ddpm_unet_base(config):
    backbone = UNet(**config.model.backbone)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_ddpm_unet_attention(config):
    backbone = UNetWithAttention(**config.model.backbone, image_size=config.dataset.image_size)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_ddpm_jit(config):
    backbone = JiT(**config.model.backbone)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_efficient_net_b2(config):
    model = EfficientNetB2(**config.model.backbone)
    return model

def create_pinn_efficient_net_b2(config) -> nn.Module:
    
    efficientnet = EfficientNetB2(**config.model.backbone)
    
    pinn_model = PhysicsInformedClassifier(
        backbone_classifier=efficientnet, 
        physics_config=config.physics_module,
        image_size=config.dataset.image_size 
    )
    
    return pinn_model

# Model Registry
MODEL_REGISTRY = {
    "ddpm_unet_base": create_ddpm_unet_base,
    "ddpm_unet_attention": create_ddpm_unet_attention,
    "ddpm_jit": create_ddpm_jit,
    "efficient_net_b2": create_efficient_net_b2,
    "pinn_efficient_net_b2": create_pinn_efficient_net_b2,
}

def get_model(config) -> nn.Module:
    name = config.model.get("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    builder_fn = MODEL_REGISTRY[name]
    return builder_fn(config)