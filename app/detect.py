"""Automatic watermark mask suggestion.

Video  - Multi-frame consistency (idea from Dekel et al., CVPR 2017): a watermark is
         the one thing whose *edges* stay put while the content moves. The per-pixel
         median of gradient magnitude over many frames cancels moving content and
         keeps static overlays. A temporal-std check rejects static background that
         is also perfectly still (tripod shots).

Image  - Single image has no temporal cue, so we use morphology: white + black
         top-hat isolates thin, high-contrast strokes (text, logos) inside a user
         region of interest. Always meant to be touched up in the editor.
"""

from __future__ import annotations

import cv2
import numpy as np


def _fill_holes(mask: np.ndarray, max_hole_frac: float = 0.02) -> np.ndarray:
    """Fill enclosed holes (letter counters in O, P, A...) that would keep watermark edges."""
    h, w = mask.shape
    inv = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and area <= max_hole_frac * h * w:
            out[labels == i] = 255
    return out


def _cleanup(mask: np.ndarray, min_area: int, max_area_frac: float, dilate: int) -> np.ndarray:
    h, w = mask.shape
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area_frac * h * w:
            out[labels == i] = 255
    out = _fill_holes(out)
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        out = cv2.dilate(out, k)
    return out


def sample_frames(path: str, n: int = 40, max_side: int = 960) -> tuple[list[np.ndarray], tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_side / max(w, h, 1))
    idxs = np.linspace(0, max(total - 1, 0), num=min(n, max(total, 1))).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                          if scale < 1 else f)
    cap.release()
    return frames, (h, w)


def detect_video_mask(path: str) -> np.ndarray | None:
    frames, (h, w) = sample_frames(path)
    if len(frames) < 3:
        return None
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    grads = []
    for g in grays:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grads.append(cv2.magnitude(gx, gy))
    grads = np.stack(grads)
    med_grad = np.median(grads, axis=0)          # static edges survive the median
    # fraction of frames where the pixel is an edge: overlays are edges ~always
    edge_thr = np.percentile(grads, 90)
    persistence = (grads > edge_thr * 0.5).mean(axis=0)

    # moving content => the image under the overlay changes => overall scene motion
    scene_motion = np.abs(np.diff(grays, axis=0)).mean()
    if scene_motion < 0.3:  # practically static video: no temporal signal
        return None

    thr = max(np.percentile(med_grad, 99), 20.0)
    cand = ((med_grad > thr * 0.6) & (persistence > 0.8)).astype(np.uint8) * 255
    small = _cleanup(cand, min_area=12, max_area_frac=0.12, dilate=4)
    if not small.any():
        return None
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def detect_image_mask(image: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, rw, rh = roi if roi else (0, 0, w, h)
    x, y = max(0, x), max(0, y)
    rw, rh = min(rw, w - x), min(rh, h - y)
    crop = image[y:y + rh, x:x + rw]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = max(9, (min(rw, rh) // 12) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    resp = cv2.max(tophat, blackhat)
    _, cand = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # in a user-picked ROI the watermark may fill most of it, so allow big blobs
    max_frac = 0.9 if roi else 0.05
    cand = _cleanup(cand, min_area=8, max_area_frac=max_frac, dilate=3)
    mask = np.zeros((h, w), np.uint8)
    mask[y:y + rh, x:x + rw] = cand
    return mask
