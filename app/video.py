"""Video pipeline: decode -> per-frame inpaint -> temporal stabilisation -> encode + remux audio.

Per-frame image inpainting flickers because each frame is hallucinated
independently. Full flow-guided video inpainting (ProPainter, E2FGVI) fixes that
but needs a GPU. On CPU we get most of the benefit with an adaptive temporal
blend: when the context ring around the hole barely changed since the previous
frame, the fill is blended with the previous fill; on motion or scene cuts the
blend switches off so nothing ghosts.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Callable

import cv2
import numpy as np

from .engines import Engine, normalize_mask
from .refine import _ring, apply_preset, refine

log = logging.getLogger(__name__)


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if info["fps"] > 240 or info["fps"] < 1:
        info["fps"] = 25.0
    return info


def process_video(src: str, dst: str, mask: np.ndarray, engine: Engine, *,
                  refine_strength: float = 0.6, preset: str = "none",
                  temporal: float = 0.6, crf: int = 18,
                  progress: Callable[[float], None] | None = None) -> None:
    info = probe(src)
    w, h = info["width"] & ~1, info["height"] & ~1  # yuv420p needs even dims
    mask = normalize_mask(mask, (h, w))
    ring = _ring(mask, 8) > 0
    hole = mask > 0

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{info['fps']:.6f}", "-i", "-",
        "-i", src,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", dst,
    ]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(src)
    total = max(info["frames"], 1)
    prev_fill = prev_ring = None
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = frame[:h, :w]
            out = engine(frame, mask)
            out = refine(frame, out, mask, refine_strength, seed=i)

            if temporal > 0 and prev_fill is not None:
                ring_now = frame[ring].astype(np.float32)
                change = float(np.abs(ring_now - prev_ring).mean())
                # change ~0 -> lean on previous fill; change >= 12 -> fresh fill only
                keep = temporal * float(np.clip(1 - change / 12.0, 0, 1))
                if keep > 0:
                    blended = out[hole].astype(np.float32) * (1 - keep) + prev_fill * keep
                    out[hole] = np.clip(blended, 0, 255).astype(np.uint8)
            prev_fill = out[hole].astype(np.float32)
            prev_ring = frame[ring].astype(np.float32)

            out = apply_preset(out, preset, for_video=True)
            enc.stdin.write(np.ascontiguousarray(out).tobytes())
            i += 1
            if progress and i % 5 == 0:
                progress(min(i / total, 0.99))
    finally:
        cap.release()
        if enc.stdin:
            enc.stdin.close()
        err = enc.stderr.read().decode(errors="ignore") if enc.stderr else ""
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}): {err[-500:]}")
    if i == 0:
        raise RuntimeError("Could not decode any frames from the video")
    if progress:
        progress(1.0)
