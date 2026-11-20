"""Inpainting engines.

Two engines share one interface: ``engine(image_bgr, mask) -> image_bgr``.

* ``fast``  - OpenCV Telea (classic fast-marching inpainting). No model, CPU-cheap,
              good for thin text / small logos on smooth backgrounds.
* ``lama``  - LaMa (Fourier-convolution inpainting, WACV 2022) via ONNX Runtime.
              Much better on textured backgrounds and large masks.

Both run through the same "crop strategy" used by IOPaint / lama-cleaner: only the
bounding box around the mask (plus context margin) is processed, so a 4K frame
with a small corner logo costs about the same as a 512px image.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

LAMA_SIZE = 512


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def normalize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return a uint8 {0,255} single-channel mask matching ``shape`` (h, w)."""
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            # works for both "white on opaque black" (canvas export) and
            # "white strokes on transparent": luminance weighted by alpha
            gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = (gray * (mask[:, :, 3].astype(np.float32) / 255.0)).astype(np.uint8)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def mask_box(mask: np.ndarray, margin: int, shape: tuple[int, int], square: bool) -> Box | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # context margin scales with the hole size: big holes need more context
    m = max(margin, int(0.5 * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = x0 - m, y0 - m, x1 + m, y1 + m
    if square:
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1 = cx - side // 2, cx - side // 2 + side
        y0, y1 = cy - side // 2, cy - side // 2 + side
    # shift inside image rather than clip, to keep the aspect ratio when possible
    if x0 < 0:
        x1, x0 = x1 - x0, 0
    if y0 < 0:
        y1, y0 = y1 - y0, 0
    if x1 > w:
        x0, x1 = max(0, x0 - (x1 - w)), w
    if y1 > h:
        y0, y1 = max(0, y0 - (y1 - h)), h
    return Box(x0, y0, x1, y1)


class Engine:
    name = "base"
    label = "base"

    def _inpaint_crop(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = normalize_mask(mask, image.shape[:2])
        box = mask_box(mask, margin=self.margin, shape=image.shape[:2], square=self.square)
        if box is None:
            return image.copy()
        crop = image[box.y0:box.y1, box.x0:box.x1]
        crop_mask = mask[box.y0:box.y1, box.x0:box.x1]
        filled = self._inpaint_crop(crop, crop_mask)
        out = image.copy()
        region = out[box.y0:box.y1, box.x0:box.x1]
        m = crop_mask > 0
        region[m] = filled[m]  # keep every unmasked pixel bit-exact
        return out

    margin = 32
    square = False


class OpenCVEngine(Engine):
    name = "fast"
    label = "Fast (OpenCV Telea, CPU)"
    margin = 24

    def _inpaint_crop(self, crop, mask):
        # radius ~ stroke thickness; thicker masks need a wider neighbourhood
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        radius = int(np.clip(dist.max() * 1.5, 3, 15))
        return cv2.inpaint(crop, mask, radius, cv2.INPAINT_TELEA)


class LamaEngine(Engine):
    name = "lama"
    label = "AI (LaMa, best quality)"
    margin = 64
    square = True

    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "0")) or os.cpu_count() or 4
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self._lock = threading.Lock()  # ORT sessions are thread-safe, but this caps RAM
        log.info("LaMa loaded from %s (inputs=%s, providers=%s)", model_path,
                 list(self.inputs), providers)

    def _inpaint_crop(self, crop, mask):
        h, w = crop.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(crop, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        m = cv2.resize(mask, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
        # slight dilation: anti-aliased watermark edges leave halos otherwise
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        img_t = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        mask_t = (m > 0).astype(np.float32)[None, None]
        names = list(self.inputs)
        img_name = next((n for n in names if "image" in n.lower()), names[0])
        mask_name = next((n for n in names if "mask" in n.lower()), names[1])
        with self._lock:
            out = self.session.run(None, {img_name: img_t, mask_name: mask_t})[0][0]
        out = out.transpose(1, 2, 0)
        if out.max() <= 1.5:  # some exports return [0,1], others [0,255]
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


class EngineRegistry:
    def __init__(self):
        self.engines: dict[str, Engine] = {"fast": OpenCVEngine()}
        model_path = os.getenv("LAMA_MODEL_PATH", "models/lama_fp32.onnx")
        if os.path.exists(model_path):
            try:
                self.engines["lama"] = LamaEngine(model_path)
            except Exception:  # noqa: BLE001 - a broken model must not take the app down
                log.exception("Failed to load LaMa model; falling back to OpenCV only")
        else:
            log.warning("LaMa model not found at %s - only the 'fast' engine is available. "
                        "Run scripts/download_model.sh to enable AI inpainting.", model_path)

    @property
    def default(self) -> str:
        return "lama" if "lama" in self.engines else "fast"

    def get(self, name: str | None) -> Engine:
        return self.engines.get(name or self.default) or self.engines[self.default]

    def describe(self) -> list[dict]:
        return [{"id": k, "label": e.label, "default": k == self.default}
                for k, e in self.engines.items()]
