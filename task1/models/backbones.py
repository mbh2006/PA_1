"""Frozen backbones for Task 1.

* ``resnet50``  - torchvision ResNet-50 (IMAGENET1K_V2), 2048-d global-average-pooled feature;
* ``vit_b16``   - torchvision ViT-B/16 (IMAGENET1K_V1), 768-d final class token;
* ``clip_vit_b32`` - OpenCLIP ViT-B-32 (pretrained='openai'), 512-d normalised
  image embedding, plus the zero-shot text classifier for the fixed prompt
  "a photo of a {class}.".

Every backbone receives its own required normalisation; the *image content* is
whatever the interventions produced on the common 224x224 RGB canvas.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Backbone:
    name = "base"
    feature_dim = 0
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    supports_zero_shot = False

    def __init__(self, device: torch.device):
        self.device = device

    # ------------------------------------------------------------- interface
    def features(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def zero_shot_logits(self, images: torch.Tensor, class_names: List[str]) -> torch.Tensor:
        raise NotImplementedError

    # --------------------------------------------------------------- helpers
    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        return (images - mean) / std

    @staticmethod
    def pil_to_tensor(images: List[Image.Image], device) -> torch.Tensor:
        arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
        return tensor


class ResNet50Backbone(Backbone):
    name = "resnet50"
    feature_dim = 2048

    def __init__(self, device):
        super().__init__(device)
        from torchvision import models
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(model.children())[:-1]).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def features(self, images):
        x = self.normalize(images)
        x = self.model(x)
        return torch.flatten(x, 1)


class ViTB16Backbone(Backbone):
    name = "vit_b16"
    feature_dim = 768

    def __init__(self, device):
        super().__init__(device)
        from torchvision import models
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def features(self, images):
        x = self.normalize(images)
        tokens = self.model._process_input(x)  # type: ignore[attr-defined]
        batch = tokens.shape[0]
        class_token = self.model.class_token.expand(batch, -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        tokens = self.model.encoder(tokens)
        return tokens[:, 0]


class CLIPViTB32Backbone(Backbone):
    name = "clip_vit_b32"
    feature_dim = 512
    mean = CLIP_MEAN
    std = CLIP_STD
    supports_zero_shot = True

    def __init__(self, device):
        super().__init__(device)
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.model = model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._text_features = None
        self._class_names: Optional[List[str]] = None

    @torch.no_grad()
    def features(self, images):
        tensor = self.normalize(images)
        embedding = self.model.encode_image(tensor)
        return F.normalize(embedding, dim=1)

    @torch.no_grad()
    def text_features(self, class_names: List[str]) -> torch.Tensor:
        if self._text_features is None or self._class_names != class_names:
            prompts = [f"a photo of a {name}." for name in class_names]
            tokens = self.tokenizer(prompts).to(self.device)
            text = self.model.encode_text(tokens)
            self._text_features = F.normalize(text, dim=1)
            self._class_names = list(class_names)
        return self._text_features

    @torch.no_grad()
    def zero_shot_logits(self, images: torch.Tensor, class_names: List[str]) -> torch.Tensor:
        image_features = self.features(images)
        text_features = self.text_features(class_names)
        scale = self.model.logit_scale.exp()
        return scale * image_features @ text_features.t()


def build_backbone(name: str, device: torch.device) -> Backbone:
    registry = {
        "resnet50": ResNet50Backbone,
        "vit_b16": ViTB16Backbone,
        "clip_vit_b32": CLIPViTB32Backbone,
    }
    if name not in registry:
        raise KeyError(f"unknown backbone '{name}'. Available: {sorted(registry)}")
    return registry[name](device)
