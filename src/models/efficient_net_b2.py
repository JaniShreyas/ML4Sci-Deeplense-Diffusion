# Use timm to create an EfficientNet-B2 model for binary classification
# And  structure as required by the training loop

import torch
import torch.nn as nn
import timm


class EfficientNetB2(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, in_channels=1):
        super(EfficientNetB2, self).__init__()
        self.model = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
    
    def sample(self, *args, **kwargs):
        raise NotImplementedError("This model is not meant for sampling.")