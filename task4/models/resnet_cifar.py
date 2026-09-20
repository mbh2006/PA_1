"""CIFAR-adapted ResNet-18 for Task 4.

Assignment requirements: replace the ImageNet 7x7 stride-2 first convolution
with a 3x3 stride-1 convolution and remove the initial max-pooling layer; train
from random initialisation (Vanilla, GCSC) and operate on 32x32 inputs.

PROSER needs two pieces of the network separately:

* ``features_pre``  - everything up to and including layer2 (manifold mixup
  happens after layer2 and before layer3);
* ``features_post`` - layer3, layer4 and global average pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CifarResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, num_dummies: int = 0, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        # CIFAR stem: 3x3 stride-1 conv, no max-pool
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.feature_dim = net.fc.in_features
        self.num_known = num_classes
        self.num_dummies = num_dummies
        net.fc = nn.Linear(self.feature_dim, num_classes + num_dummies)
        self.net = net

    # ------------------------------------------------------------------ pieces
    def features_pre(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        return x

    def features_post(self, h: torch.Tensor) -> torch.Tensor:
        net = self.net
        h = net.layer3(h)
        h = net.layer4(h)
        h = net.avgpool(h)
        return torch.flatten(h, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features_post(self.features_pre(x))

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.net.fc(features)

    def forward(self, x: torch.Tensor):
        features = self.features(x)
        logits = self.classifier(features)
        return logits, features

    # ------------------------------------------------------------------ helpers
    def known_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, :self.num_known]

    def dummy_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, self.num_known:]

    def augmented_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """[known logits, strongest dummy logit] - the K+1-way open-set output."""
        known = self.known_logits(logits)
        if self.num_dummies == 0:
            return known
        dummy = self.dummy_logits(logits).max(dim=1, keepdim=True).values
        return torch.cat([known, dummy], dim=1)
