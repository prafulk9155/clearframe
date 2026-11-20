import os
import subprocess

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.detect import detect_image_mask, detect_video_mask
from app.engines import OpenCVEngine, mask_box
from app.main import app
from app.refine import PRESETS, apply_preset, refine

client = TestClient(app)


def scene(h=360, w=640, shift=0):
    """Smooth gradient + soft blobs: a stand-in for natural content."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = xx + shift
    img = np.dstack([
        120 + 60 * np.sin(xx / 70.0),
        110 + 50 * np.cos(yy / 55.0),
        130 + 40 * np.sin((xx + yy) / 90.0),
    ])
    return np.clip(img, 0, 255).astype(np.uint8)


def stamp(img, text="SAMPLE", org=(400, 330)):
    over = img.copy()
    cv2.putText(over, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
    out = cv2.addWeighted(over, 0.6, img, 0.4, 0)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.putText(m, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 4, cv2.LINE_AA)
    return out, cv2.dilate((m > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))


def err(a, b, m):
    return float(np.abs(a.astype(int) - b.astype(int))[m > 0].mean())


def test_fast_engine_removes_watermark():
    clean = scene()
    marked, mask = stamp(clean)
    out = OpenCVEngine()(marked, mask)
    assert err(out, clean, mask) < err(marked, clean, mask) * 0.25
    # unmasked pixels must be untouched
    assert np.array_equal(out[mask == 0], marked[mask == 0])


def test_refine_keeps_outside_and_improves_seams():
    clean = scene()
    marked, mask = stamp(clean)
    raw = OpenCVEngine()(marked, mask)
    out = refine(marked, raw, mask, 0.7)
    far = cv2.dilate(mask, np.ones((31, 31), np.uint8)) == 0
    assert np.array_equal(out[far], marked[far])
    assert err(out, clean, mask) < err(marked, clean, mask) * 0.3


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_run(preset):
    out = apply_preset(scene(), preset)
    assert out.shape == (360, 640, 3) and out.dtype == np.uint8


def test_mask_box_square_and_in_bounds():
    m = np.zeros((100, 200), np.uint8)
    m[90:99, 190:199] = 255
    b = mask_box(m, 32, m.shape, square=True)
    assert 0 <= b.x0 < b.x1 <= 200 and 0 <= b.y0 < b.y1 <= 100


def test_image_detect_in_roi():
    marked, mask = stamp(scene())
    det = detect_image_mask(marked, (380, 290, 260, 60))
    inter = ((det > 0) & (mask > 0)).sum()
    assert inter / (mask > 0).sum() > 0.5


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("v") / "in.mp4")
    h, w = 360, 640
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                             "bgr24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
                             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                             "-shortest", path], stdin=subprocess.PIPE)
    for t in range(50):
        frame, _ = stamp(scene(shift=t * 6))
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return path


def test_video_detect(video_file):
    _, truth = stamp(scene())
    det = detect_video_mask(video_file)
    assert det is not None
    hit = ((det > 0) & (truth > 0)).sum() / (truth > 0).sum()
    assert hit > 0.6, hit
    assert (det > 0).mean() < 0.1


def test_api_image_roundtrip():
    marked, mask = stamp(scene())
    r = client.post("/api/image/inpaint",
                    files={"image": ("a.png", cv2.imencode(".png", marked)[1].tobytes(), "image/png"),
                           "mask": ("m.png", cv2.imencode(".png", mask)[1].tobytes(), "image/png")},
                    data={"engine": "fast", "refine_strength": "0.6", "preset": "natural"})
    assert r.status_code == 200, r.text
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == marked.shape


def test_api_rejects_empty_mask():
    img = cv2.imencode(".png", scene())[1].tobytes()
    blank = cv2.imencode(".png", np.zeros((360, 640), np.uint8))[1].tobytes()
    r = client.post("/api/image/inpaint", files={"image": ("a.png", img), "mask": ("m.png", blank)})
    assert r.status_code == 400


def test_api_video_job(video_file):
    _, mask = stamp(scene())
    with open(video_file, "rb") as f:
        r = client.post("/api/video/inpaint",
                        files={"video": ("in.mp4", f.read(), "video/mp4"),
                               "mask": ("m.png", cv2.imencode(".png", mask)[1].tobytes())},
                        data={"engine": "fast"})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    import time
    for _ in range(120):
        s = client.get(f"/api/jobs/{jid}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert s["status"] == "done", s
    res = client.get(f"/api/jobs/{jid}/result")
    assert res.status_code == 200 and len(res.content) > 1000
    out = os.path.join(os.path.dirname(video_file), "out.mp4")
    open(out, "wb").write(res.content)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                              "-of", "csv=p=0", out], capture_output=True, text=True).stdout
    assert "video" in streams and "audio" in streams


def test_normalize_mask_rgba_variants():
    from app.engines import normalize_mask
    opaque = np.zeros((10, 10, 4), np.uint8); opaque[..., 3] = 255; opaque[2:4, 2:4, :3] = 255
    transp = np.zeros((10, 10, 4), np.uint8); transp[2:4, 2:4] = 255
    for m in (opaque, transp):
        out = normalize_mask(m, (10, 10))
        assert out.sum() == 4 * 255, out
