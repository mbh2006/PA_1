"""ResNet-18 backbone for Tasks 2 and 3.

torchvision ResNet-18 with ``IMAGENET1K_V1`` weights, the ImageNet classifier
replaced by a 7-class linear head. Task 2 fine-tunes the *complete* network for
every method; the 512-d global-average-pooled feature (the input of ``fc``) is
what DAN/DANN/CDAN align.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet18PACS(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.resnet18(weights=weights)
        self.feature_dim = self.net.fc.in_features  # 512
        self.net.fc = nn.Linear(self.feature_dim, num_classes)

    # ------------------------------------------------------------------ pieces
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Final 512-d representation (global-average-pooled conv features)."""
        net = self.net
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)
        x = net.avgpool(x)
        return torch.flatten(x, 1)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.net.fc(features)

    def forward(self, x: torch.Tensor):
        features = self.features(x)
        logits = self.classifier(features)
        return logits, features
