"""Deterministic controlled interventions for Task 1.

All interventions are built on a common 224x224 RGB PIL image *before* any
model-specific normalisation, so every backbone receives identical pixels.

* grayscale  - remove colour (geometry unchanged);
* hue rotation - change colour while preserving geometry (the second colour
  intervention chosen for the report);
* translation - reflection-pad and shift-crop by 0/8/16/32 px in the four
  cardinal directions;
* patch shuffle - split into a 4x4 pixel grid and apply one non-identity
  permutation per image, seeded from a fixed base so the identical shuffled
  image is reused by every model.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image

SHUFFLE_GRID = 4
SHUFFLE_SEED = 6304


def grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L").convert("RGB")


def hue_rotate(image: Image.Image, degrees: float = 120.0) -> Image.Image:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    shift = int(round(degrees / 360.0 * 255.0)) % 255
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + shift) % 255
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def translate(image: Image.Image, pixels: int, direction: str) -> Image.Image:
    """Shift the object using reflection padding followed by a shifted crop."""
    width, height = image.size
    pad = pixels
    padded = Image.new("RGB", (width + 2 * pad, height + 2 * pad))
    padded.paste(image, (pad, pad))
    reflected = np.array(padded)  # writable copy
    if pad > 0:
        reflected[:pad, :, :] = reflected[pad:2 * pad, :, :][::-1, :, :]
        reflected[-pad:, :, :] = reflected[-2 * pad:-pad, :, :][::-1, :, :]
        reflected[:, :pad, :] = reflected[:, pad:2 * pad, :][:, ::-1, :]
        reflected[:, -pad:, :] = reflected[:, -2 * pad:-pad, :][:, ::-1, :]
    padded = Image.fromarray(reflected)
    left, top = pad, pad
    if direction == "right":
        left = pad - pixels
    elif direction == "left":
        left = pad + pixels
    elif direction == "down":
        top = pad - pixels
    elif direction == "up":
        top = pad + pixels
    else:
        raise KeyError(direction)
    return padded.crop((left, top, left + width, top + height))


def patch_permutation(image_index: int, grid: int = SHUFFLE_GRID) -> List[int]:
    """One non-identity permutation for a given image (deterministic)."""
    rng = np.random.RandomState(SHUFFLE_SEED + int(image_index))
    permutation = rng.permutation(grid * grid)
    if np.array_equal(permutation, np.arange(grid * grid)):  # cannot happen for grid>1, but be safe
        permutation[[0, 1]] = permutation[[1, 0]]
    return permutation.tolist()


def patch_shuffle(image: Image.Image, image_index: int, grid: int = SHUFFLE_GRID) -> Image.Image:
    width, height = image.size
    tile_w, tile_h = width // grid, height // grid
    permutation = patch_permutation(image_index, grid)
    output = Image.new("RGB", (width, height))
    for position, source in enumerate(permutation):
        row, column = divmod(position, grid)
        source_row, source_column = divmod(source, grid)
        tile = image.crop((source_column * tile_w, source_row * tile_h,
                           (source_column + 1) * tile_w, (source_row + 1) * tile_h))
        output.paste(tile, (column * tile_w, row * tile_h))
    return output


INTERVENTIONS = {
    "clean": lambda image, index: image,
    "grayscale": lambda image, index: grayscale(image),
    "hue": lambda image, index: hue_rotate(image, 120.0),
    "shuffle": lambda image, index: patch_shuffle(image, index),
}

TRANSLATION_DIRECTIONS = ["left", "right", "up", "down"]
TRANSLATION_DISPLACEMENTS = [0, 8, 16, 32]
