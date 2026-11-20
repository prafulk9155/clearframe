"""Colour-select mask ("magic wand" across the whole frame).

Best tool for overlays that have a distinctive colour: annotation pens, glow
doodles, coloured logos, subtitles. Two tests are OR-ed:

* LAB distance to the picked colour (lightness down-weighted for coloured
  picks, because glows vary in brightness but keep their hue)
* for coloured picks only: same hue angle at lower chroma, which catches the
  faint anti-aliased / glow halo that a pure distance test leaves behind
"""

from __future__ import annotations

import cv2
import numpy as np


def color_mask(image: np.ndarray, colors: list[tuple[int, int, int]], tolerance: float = 30,
               grow: int = 6) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    mask = np.zeros(image.shape[:2], bool)
    for rgb in colors:
        s = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        sL, sa, sb = s[0], s[1] - 128, s[2] - 128
        s_chroma = float(np.hypot(sa, sb))
        coloured = s_chroma > 20
        wL = 0.15 if coloured else 1.0
        d = np.sqrt(wL * (L - sL) ** 2 + (a - sa) ** 2 + (b - sb) ** 2)
        m = d < tolerance
        if coloured:
            s_hue = float(np.degrees(np.arctan2(sb, sa)))
            dh = np.abs((hue - s_hue + 180) % 360 - 180)
            m |= (dh < 0.6 * tolerance) & (chroma > max(10.0, 0.3 * s_chroma))
        mask |= m
    out = mask.astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop speckle
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        out = cv2.dilate(out, k)
    return out
