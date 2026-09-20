"""Cue-conflict generation: AdaIN-style transfer as a small optimisation.

Implements the AdaIN objective (Huang & Belongie, 2017) directly on pixels:
the stylised image is optimised so that its VGG-19 relu4_1 features match the
content image's structure while the channel-wise means/stds of its layers
relu1_1..relu4_1 match the style image. This needs no pretrained decoder, is
deterministic given a seed, and preserves the content layout - which is exactly
what a cue conflict needs: shape from the content class, texture from the style
class.

``make_cue_conflicts.py`` applies a model-free rejection rule to the outputs.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models

CONTENT_LAYER = "relu4_1"
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
_LAYER_INDICES = {"relu1_1": 1, "relu2_1": 6, "relu3_1": 11, "relu4_1": 20}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_vgg19(device: torch.device):
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    features = vgg.features.eval().to(device)
    for parameter in features.parameters():
        parameter.requires_grad_(False)
    return features


def _to_tensor(image: Image.Image, device, size: int = 224) -> torch.Tensor:
    resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    array = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
    tensor = array.unsqueeze(0).to(device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (tensor - mean) / std


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    array = (tensor * std + mean).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def _features(vgg, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """VGG features at the requested layers, computed incrementally.

    Gradients flow through ``tensor`` (needed for the stylisation optimisation);
    the VGG weights themselves are frozen, so no parameter gradients accumulate.
    """
    out: Dict[str, torch.Tensor] = {}
    x = tensor
    previous = -1
    for name, index in sorted(_LAYER_INDICES.items(), key=lambda kv: kv[1]):
        x = vgg[previous + 1: index + 1](x)
        out[name] = x
        previous = index
    return out


def _statistics(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=(2, 3))
    std = features.std(dim=(2, 3), unbiased=False)
    return mean, std


def adain_transfer(content: Image.Image, style: Image.Image, vgg, device,
                   steps: int = 150, lr: float = 0.05, seed: int = 6304,
                   content_weight: float = 1.0, style_weight: float = 1e2) -> Image.Image:
    """Return the stylised content image (PIL, 224x224)."""
    torch.manual_seed(seed)
    content_tensor = _to_tensor(content, device)
    style_tensor = _to_tensor(style, device)

    with torch.no_grad():
        content_features = _features(vgg, content_tensor)[CONTENT_LAYER].detach()
        style_stats = {}
        for name, features in _features(vgg, style_tensor).items():
            mean, std = _statistics(features)
            style_stats[name] = (mean.detach(), std.detach())
        minimum = float(((0.0 - IMAGENET_MEAN) / IMAGENET_STD).min())
        maximum = float(((1.0 - IMAGENET_MEAN) / IMAGENET_STD).max())

    stylised = content_tensor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([stylised], lr=lr)
    for _ in range(steps):
        features = _features(vgg, stylised)
        content_loss = F.mse_loss(features[CONTENT_LAYER], content_features)
        style_loss = stylised.new_zeros(())
        for layer in STYLE_LAYERS:
            mean, std = _statistics(features[layer])
            target_mean, target_std = style_stats[layer]
            style_loss = style_loss + F.mse_loss(mean, target_mean) + F.mse_loss(std, target_std)
        loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            stylised.clamp_(minimum, maximum)

    with torch.no_grad():
        return _to_pil(stylised.detach())
