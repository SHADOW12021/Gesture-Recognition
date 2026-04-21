from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_model(architecture: str, num_classes: int) -> nn.Module:
    architecture = architecture.lower()

    if architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "squeezenet1_1":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")
