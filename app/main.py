"""ClearFrame API + static web app."""

from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .colormask import color_mask
from .detect import detect_image_mask, detect_video_mask
from .engines import EngineRegistry, normalize_mask
from .refine import PRESETS, apply_preset, refine
from .video import probe, process_video

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clearframe")

MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "25"))
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "300"))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "600"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "6000"))
JOB_TTL = int(os.getenv("JOB_TTL_SECONDS", "3600"))
WORKERS = int(os.getenv("VIDEO_WORKERS", "1"))
WORK_DIR = Path(os.getenv("WORK_DIR", tempfile.gettempdir())) / "clearframe"
WORK_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="ClearFrame", version="1.0.0")
engines = EngineRegistry()
pool = ThreadPoolExecutor(max_workers=WORKERS)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ---------- helpers ----------
async def _read_limited(f: UploadFile, max_mb: int) -> bytes:
    data = await f.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {max_mb} MB)")
    if not data:
        raise HTTPException(400, "Empty upload")
    return data


def _decode(data: bytes, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if img is None:
        raise HTTPException(415, "Unsupported or corrupt image")
    return img


def _png_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _gc_jobs():
    now = time.time()
    with jobs_lock:
        for jid in [j for j, v in jobs.items() if now - v["created"] > JOB_TTL]:
            shutil.rmtree(jobs[jid]["dir"], ignore_errors=True)
            jobs.pop(jid, None)


# ---------- API ----------
@app.get("/api/health")
def health():
    return {"ok": True, "engines": engines.describe(), "presets": list(PRESETS),
            "limits": {"image_mb": MAX_IMAGE_MB, "video_mb": MAX_VIDEO_MB,
                       "video_seconds": MAX_VIDEO_SECONDS}}


@app.post("/api/image/inpaint")
async def image_inpaint(image: UploadFile = File(...), mask: UploadFile = File(...),
                        engine: str = Form(None), refine_strength: float = Form(0.6),
                        preset: str = Form("none"), fmt: str = Form("png")):
    raw = await _read_limited(image, MAX_IMAGE_MB)
    img = _decode(raw, cv2.IMREAD_UNCHANGED)
    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha, img = img[:, :, 3], img[:, :, :3].copy()
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(float(img.max()), 1.0))
    if max(img.shape[:2]) > MAX_IMAGE_SIDE:
        raise HTTPException(413, f"Image too large (max side {MAX_IMAGE_SIDE}px)")
    m = normalize_mask(_decode(await _read_limited(mask, MAX_IMAGE_MB), cv2.IMREAD_UNCHANGED),
                       img.shape[:2])
    if not m.any():
        raise HTTPException(400, "Mask is empty - paint over the watermark first")

    t0 = time.perf_counter()
    eng = engines.get(engine)
    out = eng(img, m)
    out = refine(img, out, m, refine_strength)
    out = apply_preset(out, preset)
    if alpha is not None and fmt == "png":
        out = np.dstack([out, alpha])
    ext, mime, params = (".jpg", "image/jpeg", [cv2.IMWRITE_JPEG_QUALITY, 95]) if fmt == "jpg" \
        else (".png", "image/png", [])
    ok, buf = cv2.imencode(ext, out, params)
    ms = int((time.perf_counter() - t0) * 1000)
    log.info("image %sx%s engine=%s %dms", img.shape[1], img.shape[0], eng.name, ms)
    return Response(buf.tobytes(), media_type=mime,
                    headers={"X-Engine": eng.name, "X-Process-Ms": str(ms)})


@app.post("/api/image/detect")
async def image_detect(image: UploadFile = File(...), x: int = Form(None), y: int = Form(None),
                       w: int = Form(None), h: int = Form(None)):
    img = _decode(await _read_limited(image, MAX_IMAGE_MB))
    roi = (x, y, w, h) if None not in (x, y, w, h) and w > 4 and h > 4 else None
    mask = detect_image_mask(img, roi)
    return {"mask": _png_data_url(mask), "coverage": float((mask > 0).mean())}


VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def _video_suffix(name: str | None) -> str:
    suffix = Path(name or "").suffix.lower()
    return suffix if suffix in VIDEO_EXTS else ".mp4"


@app.post("/api/image/color-mask")
async def image_color_mask(image: UploadFile = File(...), colors: str = Form(...),
                           tolerance: float = Form(30), grow: int = Form(4)):
    """colors: '#rrggbb,#rrggbb' picked by the user; returns a mask of all similar pixels."""
    img = _decode(await _read_limited(image, MAX_IMAGE_MB))
    try:
        rgb = [tuple(int(c.strip().lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
               for c in colors.split(",") if c.strip()][:8]
    except ValueError:
        raise HTTPException(400, "colors must be like #ff8800,#ffffff")
    if not rgb:
        raise HTTPException(400, "Pick at least one colour")
    mask = color_mask(img, rgb, float(min(max(tolerance, 5), 80)), int(min(max(grow, 0), 20)))
    return {"mask": _png_data_url(mask), "coverage": float((mask > 0).mean())}


def _save_upload(data: bytes, suffix: str) -> tuple[str, Path]:
    jid = uuid.uuid4().hex[:16]
    d = WORK_DIR / jid
    d.mkdir(parents=True)
    src = d / f"input{suffix}"
    src.write_bytes(data)
    return jid, d


@app.post("/api/video/detect")
async def video_detect(video: UploadFile = File(...)):
    data = await _read_limited(video, MAX_VIDEO_MB)
    suffix = _video_suffix(video.filename)
    _, d = _save_upload(data, suffix)
    try:
        mask = detect_video_mask(str(d / f"input{suffix}"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if mask is None:
        return {"mask": None, "message": "No static overlay found - paint the mask manually."}
    return {"mask": _png_data_url(mask), "coverage": float((mask > 0).mean())}


@app.post("/api/video/inpaint")
async def video_inpaint(video: UploadFile = File(...), mask: UploadFile = File(...),
                        engine: str = Form(None), refine_strength: float = Form(0.6),
                        preset: str = Form("none"), temporal: float = Form(0.6)):
    _gc_jobs()
    data = await _read_limited(video, MAX_VIDEO_MB)
    suffix = _video_suffix(video.filename)
    jid, d = _save_upload(data, suffix)
    src = str(d / f"input{suffix}")
    info = probe(src)
    if info["width"] == 0 or info["frames"] == 0:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(415, "Unsupported or corrupt video")
    if info["frames"] / info["fps"] > MAX_VIDEO_SECONDS:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(413, f"Video too long (max {MAX_VIDEO_SECONDS}s)")
    m = _decode(await _read_limited(mask, 20), cv2.IMREAD_UNCHANGED)
    if not normalize_mask(m, (info["height"], info["width"])).any():
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "Mask is empty - paint over the watermark first")

    job = {"id": jid, "status": "queued", "progress": 0.0, "error": None,
           "created": time.time(), "dir": str(d), "info": info,
           "out": str(d / "output.mp4")}
    with jobs_lock:
        jobs[jid] = job

    def run():
        job["status"] = "processing"
        t0 = time.perf_counter()
        try:
            process_video(src, job["out"], m, engines.get(engine),
                          refine_strength=refine_strength, preset=preset, temporal=temporal,
                          progress=lambda p: job.__setitem__("progress", round(p, 3)))
            job["status"] = "done"
            job["seconds"] = round(time.perf_counter() - t0, 1)
        except Exception as e:  # noqa: BLE001
            log.exception("video job %s failed", jid)
            job["status"], job["error"] = "error", str(e)[:300]

    pool.submit(run)
    return {"job_id": jid, "info": info}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Job not found or expired")
    return {k: job.get(k) for k in ("id", "status", "progress", "error", "info", "seconds")}


@app.get("/api/jobs/{jid}/result")
def job_result(jid: str):
    job = jobs.get(jid)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    return FileResponse(job["out"], media_type="video/mp4", filename=f"clearframe-{jid}.mp4")


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse({"detail": "Internal error"}, status_code=500)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
