# How watermark removers actually work (research notes)

Every "AI watermark remover" on the web is the same two-stage pipeline:

```
input ──► 1. MASK (where is the mark?) ──► 2. INPAINT (fill the hole) ──► 3. REFINE ──► output
```

The differences between products are only in *how good* each stage is and
whether they run on GPU servers or in the browser.

## Stage 1 — Mask

| Approach | Used by | Notes |
|---|---|---|
| Manual brush / box | IOPaint, most web tools | Most reliable; always offered as fallback |
| Morphological top-hat | Classic CV | Finds thin bright/dark strokes (text). Good inside a user ROI |
| **Multi-frame consistency** | Dekel et al., CVPR 2017 (Google) | A watermark is identical across many images/frames while content changes. Median of gradients over the collection isolates it |
| Segmentation models (SAM, YOLO-seg, watermark detectors) | Commercial tools | Best automatic results; heavy |

ClearFrame implements the first three: brush/box, top-hat in ROI for images,
and gradient-median + edge-persistence for video (the multi-frame idea applied to
frames instead of a stock-photo collection).

## Stage 2 — Inpainting

| Model | Year | Quality | Cost |
|---|---|---|---|
| OpenCV Telea / Navier–Stokes | 2004 | OK for thin text on smooth areas; blurs texture | ~ms on CPU |
| **LaMa** (Fourier convolutions) | WACV 2022 | Excellent for watermarks; global receptive field handles big holes & repeated texture | ~0.5–2 s / 512² on CPU, ~50 ms GPU |
| Stable Diffusion inpaint / SDXL | 2022+ | Can hallucinate new content; overkill & slower for watermarks | GPU required |
| ProPainter / E2FGVI (video) | 2022–23 | Flow-guided propagation — copies real pixels from other frames; temporally stable | GPU required |

LaMa is the de-facto engine behind IOPaint / lama-cleaner and most "remove
watermark" sites. ClearFrame ships it as an ONNX model (Apache-2.0, ~208 MB).

**Crop strategy (from IOPaint):** run the model only on a square crop around
the mask with generous context, resize to 512, inpaint, resize back, and paste
*only the masked pixels*. This keeps 4K frames fast and leaves every
unmasked pixel bit-exact.

## Stage 3 — Refinement (what separates "pro" output from obvious edits)

1. **Feathered composite** — soft alpha at the mask border removes seams.
2. **Grain synthesis** — inpainted regions are unnaturally clean. Measure noise
   (MAD of high-pass) in a ring around the hole and re-inject it.
3. **Colour transfer** — LAB mean/std matching from the ring corrects drift.
4. **Edge-preserving smoothing** (bilateral) inside the hole removes model artefacts.

## Video specifics

Frame-by-frame image inpainting flickers. Options in order of quality:

1. Flow-guided video inpainting (ProPainter) — best; GPU only.
2. **Adaptive temporal blend** (ClearFrame default on CPU) — blend each fill with
   the previous one, weighted by how much the surrounding ring changed. Static
   shots become rock-steady; on motion/cuts the blend switches off (no ghosting).
3. Plain per-frame — flickers.

## Semi-transparent marks: reverse alpha blending

A visible watermark is `I = α·W + (1−α)·B`. If `W` and `α` are known (estimated
from many frames/images, as in Dekel et al.), the original is recovered exactly:
`B = (I − α·W) / (1 − α)` — no hallucination at all. This is the best possible
result for semi-transparent overlays and is the main item on the roadmap.

## References

- Suvorov et al., *Resolution-robust Large Mask Inpainting with Fourier Convolutions* (LaMa), WACV 2022
- Dekel, Rubinstein, Liu, Freeman, *On the Effectiveness of Visible Watermarks*, CVPR 2017
- Zhou et al., *ProPainter: Improving Propagation and Transformer for Video Inpainting*, ICCV 2023
- Telea, *An Image Inpainting Technique Based on the Fast Marching Method*, 2004
- IOPaint (formerly lama-cleaner) — open-source reference implementation of the crop strategy
