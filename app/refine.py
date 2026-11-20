"""Post-inpainting refinement and output filters.

Raw inpainting output has three typical tells, which this module fixes:

1. Hard seams at the mask border      -> feathered alpha composite
2. "Plastic" smooth fill (no grain)   -> re-synthesise sensor noise measured
                                         in a ring around the hole
3. Slight colour/brightness drift     -> mean/std transfer in LAB from the ring

Then an optional global look ("preset") is applied to the whole frame.
"""

from __future__ import annotations

import cv2
import numpy as np

PRESETS = ("none", "natural", "crisp", "vivid", "matte")


def _ring(mask: np.ndarray, width: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return cv2.dilate(mask, k) & ~cv2.dilate(mask, np.ones((3, 3), np.uint8))


def _color_match(result, mask_b, ring_b, amount):
    if ring_b.sum() < 50 or mask_b.sum() < 10:
        return result
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        ch = lab[:, :, c]
        r_mu, r_sd = ch[ring_b].mean(), ch[ring_b].std() + 1e-3
        h_mu, h_sd = ch[mask_b].mean(), ch[mask_b].std() + 1e-3
        matched = (ch[mask_b] - h_mu) * np.clip(r_sd / h_sd, 0.5, 2.0) + r_mu
        ch[mask_b] = ch[mask_b] * (1 - amount) + matched * amount
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _grain(result, original, mask_b, ring_b, amount, seed=0):
    if ring_b.sum() < 50:
        return result
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    sigma = float(np.median(np.abs(hf[ring_b]))) * 1.4826  # robust (MAD) noise estimate
    if sigma < 0.3:
        return result
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma * amount, result.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
    out = result.astype(np.float32)
    out[mask_b] += noise[mask_b][:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def refine(original: np.ndarray, result: np.ndarray, mask: np.ndarray,
           strength: float = 0.6, seed: int = 0) -> np.ndarray:
    """Blend ``result`` into ``original`` inside ``mask`` with seam/grain/colour fixes."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0 or not mask.any():
        return result
    mask_b = mask > 0
    ring_b = _ring(mask, width=max(4, int(6 + 10 * strength))) > 0

    out = result
    # edge-preserving cleanup of inpaint artefacts, only inside the hole
    smooth = cv2.bilateralFilter(out, d=5, sigmaColor=20 + 30 * strength, sigmaSpace=5)
    out = out.copy()
    out[mask_b] = smooth[mask_b]
    out = _color_match(out, mask_b, ring_b, amount=0.5 * strength)
    out = _grain(out, original, mask_b, ring_b, amount=strength, seed=seed)

    # feathered composite so the boundary has no visible seam
    feather = 1 + 2 * int(1 + 4 * strength)
    soft = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(soft, (feather, feather), 0)
    soft = np.maximum(soft, mask_b.astype(np.float32))[:, :, None]
    blended = out.astype(np.float32) * soft + original.astype(np.float32) * (1 - soft)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _unsharp(img, amount, radius):
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def apply_preset(img: np.ndarray, preset: str, for_video: bool = False) -> np.ndarray:
    if preset in (None, "", "none"):
        return img
    if preset == "natural":
        # NL-means is too slow per-frame; video gets a light bilateral instead
        if for_video:
            img = cv2.bilateralFilter(img, 5, 12, 5)
        elif img.size < 4e6 * 3:
            img = cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 5, 15)
        return _unsharp(img, 0.25, 1.0)
    if preset == "crisp":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
        return _unsharp(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), 0.6, 1.2)
    if preset == "vivid":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.convertScaleAbs(img, alpha=1.06, beta=-4)
    if preset == "matte":
        lut = np.clip(np.arange(256) * 0.88 + 18, 0, 255).astype(np.uint8)
        return cv2.LUT(img, lut)
    return img
