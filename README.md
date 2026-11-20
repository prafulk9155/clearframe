# ClearFrame

Remove watermarks, logos, timestamps and unwanted objects from **images and videos** in the browser.
Paint over the mark (or let auto-detect find it), pick an engine, and download a refined, filtered result.

- **Two engines** — `fast` (OpenCV Telea, no model) and `lama` (LaMa AI inpainting via ONNX Runtime, CPU or GPU)
- **Auto-detect** — multi-frame consistency for static video overlays; top-hat morphology inside a box for images
- **Refinement** — feathered seams, grain re-synthesis, LAB colour matching, edge-preserving cleanup
- **Output filters** — Natural / Crisp / Vivid / Matte
- **Video** — background jobs with progress, adaptive temporal smoothing (anti-flicker), audio kept, H.264 + faststart
- **Before/after slider**, undo, brush/box/eraser, keyboard shortcuts (`B` `R` `E` `[` `]` `Ctrl+Z`), paste from clipboard

How it works and why each piece was chosen: [docs/RESEARCH.md](docs/RESEARCH.md).

> Use only on media you own or are licensed to edit. Removing a rights holder's watermark can infringe copyright.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # also needs ffmpeg on PATH
bash scripts/download_model.sh           # optional: enables the LaMa AI engine (~208 MB)
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Run with Docker

```bash
docker compose up -d --build                         # http://SERVER:8000
DOMAIN=erase.example.com docker compose --profile https up -d --build   # auto-HTTPS via Caddy
```

Build without the model (smaller image, fast engine only): `docker build --build-arg WITH_LAMA=0 .`

### Deploy on an Oracle Cloud Always-Free VM (Ampere ARM)

All dependencies have arm64 wheels, so build on the VM itself:

```bash
sudo apt-get install -y docker.io docker-compose-v2 git
git clone https://github.com/<you>/clearframe && cd clearframe
DOMAIN=erase.example.com sudo -E docker compose --profile https up -d --build
```

Open ports 80/443 in both the VCN security list and the VM firewall
(`sudo iptables -I INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT`).
If you front it with your own nginx instead, set `client_max_body_size 300m;` and `proxy_read_timeout 300s;`.

## API

| Method | Path | Body (multipart) | Returns |
|---|---|---|---|
| GET | `/api/health` | – | engines, presets, limits |
| POST | `/api/image/inpaint` | `image`, `mask` (PNG, white = remove), `engine`, `refine_strength` 0–1, `preset`, `fmt` png\|jpg | image bytes |
| POST | `/api/image/detect` | `image`, optional ROI `x y w h` | `{mask: dataURL}` |
| POST | `/api/video/detect` | `video` | `{mask: dataURL \| null}` |
| POST | `/api/video/inpaint` | `video`, `mask`, `engine`, `refine_strength`, `preset`, `temporal` 0–0.9 | `{job_id}` |
| GET | `/api/jobs/{id}` | – | status, progress |
| GET | `/api/jobs/{id}/result` | – | MP4 |

```bash
curl -F image=@photo.jpg -F mask=@mask.png -F engine=lama -F preset=natural \
     http://localhost:8000/api/image/inpaint -o clean.png
```

## Configuration (env vars)

| Var | Default | |
|---|---|---|
| `LAMA_MODEL_PATH` | `models/lama_fp32.onnx` | LaMa ONNX model; app falls back to `fast` if missing |
| `MAX_IMAGE_MB` / `MAX_IMAGE_SIDE` | 25 / 6000 | image limits |
| `MAX_VIDEO_MB` / `MAX_VIDEO_SECONDS` | 300 / 600 | video limits |
| `VIDEO_WORKERS` | 1 | concurrent video jobs |
| `JOB_TTL_SECONDS` | 3600 | finished jobs + temp files are purged after this |
| `ORT_THREADS` | all cores | ONNX Runtime intra-op threads |

## Architecture

```
static/ (vanilla JS, no build)      app/
  canvas mask editor  ──HTTP──►   main.py     FastAPI routes, limits, job queue
  compare slider                  engines.py  OpenCV + LaMa (crop strategy)
  job polling                     detect.py   auto mask (video multi-frame / image top-hat)
                                  refine.py   seams, grain, colour, presets
                                  video.py    cv2 decode → inpaint → temporal blend → ffmpeg pipe (+audio)
```

## Tests

```bash
pip install pytest httpx && pytest -q
```
Synthetic watermark fixtures check: masked error drops >4×, unmasked pixels stay bit-exact,
video auto-detect hits >60% of the watermark, video jobs return MP4 with audio.

## Roadmap

1. **Reverse alpha-blending** for semi-transparent marks (exact recovery, no hallucination) — see research notes
2. ProPainter GPU worker for flow-guided video inpainting
3. Redis/RQ job queue + object storage for horizontal scaling
4. Moving-watermark tracking (per-frame masks)
5. Client-side LaMa via onnxruntime-web/WebGPU for zero-upload image mode

## License

MIT for the code. LaMa model weights: Apache-2.0 (advimman/lama, ONNX export by Carve).
