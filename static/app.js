"use strict";
const $ = (id) => document.getElementById(id);

const S = {
  mode: "image", file: null, W: 0, H: 0, base: null, tool: "brush", size: 28,
  drawing: false, last: null, rectStart: null, rectNow: null, undo: [],
  objectURLs: [], pollTimer: null,
};
const view = $("view"), vctx = view.getContext("2d");
const mask = document.createElement("canvas"), mctx = mask.getContext("2d");
const tint = document.createElement("canvas"), tctx = tint.getContext("2d");

// ---------- boot ----------
fetch("/api/health").then((r) => r.json()).then((h) => {
  const sel = $("engine");
  sel.innerHTML = h.engines.map((e) => `<option value="${e.id}" ${e.default ? "selected" : ""}>${e.label}</option>`).join("");
  $("engineStatus").textContent = h.engines.some((e) => e.id === "lama")
    ? "AI engine ready" : "Fast engine only (AI model not installed)";
}).catch(() => { $("engineStatus").textContent = "API offline"; });

// ---------- mode tabs ----------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  document.querySelectorAll(".video-only").forEach((e) => e.classList.toggle("hidden", mode !== "video"));
  document.querySelectorAll(".image-only").forEach((e) => e.classList.toggle("hidden", mode !== "image"));
  $("fileInput").accept = mode === "image" ? "image/*" : "video/*";
  $("dropHint").textContent = mode === "image" ? "PNG, JPG, WEBP" : "MP4, MOV, WEBM — static watermark/logo";
  document.querySelector(".drop-inner strong").textContent = mode === "image" ? "Drop an image" : "Drop a video";
  $("runBtn").textContent = mode === "image" ? "Remove watermark" : "Process video";
  reset();
}

// ---------- file loading ----------
const drop = $("drop");
$("fileInput").addEventListener("change", (e) => e.target.files[0] && load(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && load(e.dataTransfer.files[0]));
window.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])][0];
  if (f && f.type.startsWith("image/")) { setMode("image"); load(f); }
});

function track(url) { S.objectURLs.push(url); return url; }

async function load(file) {
  const isVideo = file.type.startsWith("video/");
  if (isVideo !== (S.mode === "video")) setMode(isVideo ? "video" : "image");
  S.file = file;
  showError("");
  try {
    S.base = isVideo ? await firstFrame(file) : await loadImage(track(URL.createObjectURL(file)));
  } catch (err) { return showError("Could not read this file: " + err.message); }
  S.W = S.base.naturalWidth || S.base.width;
  S.H = S.base.naturalHeight || S.base.height;
  for (const c of [view, mask, tint]) { c.width = S.W; c.height = S.H; }
  mctx.clearRect(0, 0, S.W, S.H);
  S.undo = [];
  drop.classList.add("hidden");
  $("editor").classList.remove("hidden");
  showEditor();
  render();
  updateRun();
}

function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error("decode failed")); i.src = src; });
}

function firstFrame(file) {
  return new Promise((res, rej) => {
    const v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.preload = "auto";
    v.src = track(URL.createObjectURL(file));
    v.onerror = () => rej(new Error("browser cannot decode this video"));
    v.onloadeddata = () => { v.currentTime = Math.min(0.1, (v.duration || 1) / 2); };
    v.onseeked = () => {
      const c = document.createElement("canvas");
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext("2d").drawImage(v, 0, 0);
      res(c);
    };
  });
}

// ---------- rendering ----------
let raf = 0;
function render() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    vctx.globalAlpha = 1;
    vctx.drawImage(S.base, 0, 0);
    tctx.globalCompositeOperation = "source-over";
    tctx.clearRect(0, 0, S.W, S.H);
    tctx.drawImage(mask, 0, 0);
    tctx.globalCompositeOperation = "source-in";
    tctx.fillStyle = "#ff3d6e";
    tctx.fillRect(0, 0, S.W, S.H);
    vctx.globalAlpha = 0.55;
    vctx.drawImage(tint, 0, 0);
    vctx.globalAlpha = 1;
    if (S.rectStart && S.rectNow) {
      const r = rectOf(S.rectStart, S.rectNow);
      vctx.setLineDash([8 * scale(), 6 * scale()]);
      vctx.lineWidth = 2 * scale();
      vctx.strokeStyle = S.tool === "erase" ? "#fff" : "#ff3d6e";
      vctx.strokeRect(r.x, r.y, r.w, r.h);
      vctx.setLineDash([]);
    }
  });
}
const scale = () => S.W / (view.clientWidth || S.W);

// ---------- drawing ----------
document.querySelectorAll(".tool").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
function setTool(t) { S.tool = t; document.querySelectorAll(".tool").forEach((b) => b.classList.toggle("active", b.dataset.tool === t)); }
$("brushSize").addEventListener("input", (e) => { S.size = +e.target.value; $("brushVal").textContent = S.size; });

function pt(e) {
  const r = view.getBoundingClientRect();
  return { x: (e.clientX - r.left) * (S.W / r.width), y: (e.clientY - r.top) * (S.H / r.height) };
}
function rectOf(a, b) { return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) }; }
function pushUndo() {
  S.undo.push(mctx.getImageData(0, 0, S.W, S.H));
  if (S.undo.length > 12) S.undo.shift();
}
function stroke(a, b) {
  mctx.globalCompositeOperation = S.tool === "erase" ? "destination-out" : "source-over";
  mctx.strokeStyle = mctx.fillStyle = "#fff";
  mctx.lineCap = mctx.lineJoin = "round";
  mctx.lineWidth = S.size * scale();
  mctx.beginPath(); mctx.moveTo(a.x, a.y); mctx.lineTo(b.x, b.y); mctx.stroke();
}

view.addEventListener("pointerdown", (e) => {
  if (!S.base) return;
  view.setPointerCapture(e.pointerId);
  pushUndo();
  S.drawing = true;
  const p = pt(e);
  if (S.tool === "rect" || (S.tool === "erase" && e.shiftKey)) { S.rectStart = p; S.rectNow = p; }
  else { S.last = p; stroke(p, p); }
  render();
});
view.addEventListener("pointermove", (e) => {
  if (!S.drawing) return;
  const p = pt(e);
  if (S.rectStart) S.rectNow = p;
  else { stroke(S.last, p); S.last = p; }
  render();
});
const endDraw = () => {
  if (!S.drawing) return;
  S.drawing = false;
  if (S.rectStart && S.rectNow) {
    const r = rectOf(S.rectStart, S.rectNow);
    mctx.globalCompositeOperation = S.tool === "erase" ? "destination-out" : "source-over";
    mctx.fillStyle = "#fff";
    if (r.w > 2 && r.h > 2) mctx.fillRect(r.x, r.y, r.w, r.h);
  }
  S.rectStart = S.rectNow = null;
  render(); updateRun();
};
view.addEventListener("pointerup", endDraw);
view.addEventListener("pointercancel", endDraw);

$("undoBtn").addEventListener("click", undo);
function undo() { const d = S.undo.pop(); if (d) { mctx.putImageData(d, 0, 0); render(); updateRun(); } }
$("clearBtn").addEventListener("click", () => { pushUndo(); mctx.clearRect(0, 0, S.W, S.H); render(); updateRun(); });
$("newBtn").addEventListener("click", reset);
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); undo(); }
  else if (e.key === "b") setTool("brush");
  else if (e.key === "r") setTool("rect");
  else if (e.key === "e") setTool("erase");
  else if (e.key === "[") { $("brushSize").value = S.size = Math.max(4, S.size - 4); $("brushVal").textContent = S.size; }
  else if (e.key === "]") { $("brushSize").value = S.size = Math.min(160, S.size + 4); $("brushVal").textContent = S.size; }
});

function maskBBox() {
  const d = mctx.getImageData(0, 0, S.W, S.H).data;
  let x0 = S.W, y0 = S.H, x1 = -1, y1 = -1;
  for (let y = 0; y < S.H; y++) for (let x = 0; x < S.W; x++) {
    if (d[(y * S.W + x) * 4 + 3] > 0) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}
function hasMask() { return !!S.base && S.W > 0 && !!maskBBox(); }
function updateRun() { $("runBtn").disabled = !S.file || !hasMask(); }

function maskBlob() {
  const c = document.createElement("canvas"); c.width = S.W; c.height = S.H;
  const x = c.getContext("2d");
  x.fillStyle = "#000"; x.fillRect(0, 0, S.W, S.H); x.drawImage(mask, 0, 0);
  return new Promise((r) => c.toBlob(r, "image/png"));
}

async function applyMaskURL(url) {
  const img = await loadImage(url);
  const c = document.createElement("canvas"); c.width = S.W; c.height = S.H;
  const x = c.getContext("2d"); x.drawImage(img, 0, 0, S.W, S.H);
  const id = x.getImageData(0, 0, S.W, S.H);
  for (let i = 0; i < id.data.length; i += 4) { const v = id.data[i]; id.data[i + 3] = v > 127 ? 255 : 0; id.data[i] = id.data[i + 1] = id.data[i + 2] = 255; }
  pushUndo();
  mctx.globalCompositeOperation = "source-over";
  mctx.putImageData(id, 0, 0);
  render(); updateRun();
}

// ---------- auto-detect ----------
$("autoBtn").addEventListener("click", async () => {
  if (!S.file) return;
  const btn = $("autoBtn"); btn.disabled = true; btn.textContent = "Detecting…"; showError("");
  try {
    const fd = new FormData();
    let url;
    if (S.mode === "image") {
      fd.append("image", S.file);
      const box = maskBBox(); // a rough box/scribble = region to search in
      if (box) { fd.append("x", box.x); fd.append("y", box.y); fd.append("w", box.w); fd.append("h", box.h); }
      url = "/api/image/detect";
    } else { fd.append("video", S.file); url = "/api/video/detect"; }
    const j = await api(url, fd).then((r) => r.json());
    if (!j.mask) showError(j.message || "Nothing detected — paint the mask manually.");
    else await applyMaskURL(j.mask);
  } catch (e) { showError(e.message); }
  finally { btn.disabled = false; btn.textContent = "✨ Auto-detect"; }
});

// ---------- processing ----------
["refine", "temporal"].forEach((id) => $(id).addEventListener("input", (e) => { $(id + "Val").textContent = (+e.target.value).toFixed(2); }));

async function api(url, body) {
  const r = await fetch(url, { method: "POST", body });
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch {} throw new Error(m); }
  return r;
}

$("runBtn").addEventListener("click", async () => {
  showError(""); setBusy(true);
  const fd = new FormData();
  fd.append(S.mode === "image" ? "image" : "video", S.file);
  fd.append("mask", await maskBlob(), "mask.png");
  fd.append("engine", $("engine").value);
  fd.append("refine_strength", $("refine").value);
  fd.append("preset", $("preset").value);
  try {
    if (S.mode === "image") {
      fd.append("fmt", $("fmt").value);
      progress(0.5, "Inpainting…");
      const r = await api("/api/image/inpaint", fd);
      const url = track(URL.createObjectURL(await r.blob()));
      showImageResult(url, `${r.headers.get("X-Engine")} · ${r.headers.get("X-Process-Ms")} ms`);
    } else {
      fd.append("temporal", $("temporal").value);
      progress(0, "Uploading…");
      const { job_id } = await api("/api/video/inpaint", fd).then((r) => r.json());
      await pollJob(job_id);
    }
  } catch (e) { showError(e.message); setBusy(false); }
});

function pollJob(id) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const j = await fetch(`/api/jobs/${id}`).then((r) => r.json());
        if (j.status === "error") return reject(new Error(j.error || "Processing failed"));
        if (j.status === "done") { showVideoResult(`/api/jobs/${id}/result`, j.seconds); return resolve(); }
        progress(j.progress, j.status === "queued" ? "Queued…" : `Processing ${Math.round(j.progress * 100)}%`);
        S.pollTimer = setTimeout(tick, 1000);
      } catch (e) { reject(e); }
    };
    tick();
  });
}

function showImageResult(url, meta) {
  setBusy(false);
  const beforeURL = track(URL.createObjectURL(S.file));
  $("beforeImg").src = beforeURL; $("afterImg").src = url;
  view.classList.add("hidden"); $("compare").classList.remove("hidden");
  setCompare(50);
  const ext = $("fmt").value;
  $("downloadBtn").href = url;
  $("downloadBtn").download = (S.file.name.replace(/\.[^.]+$/, "") || "image") + `-clean.${ext}`;
  $("resultActions").classList.remove("hidden");
  $("hint").textContent = `Done (${meta}). Drag across the image to compare.`;
}
function showVideoResult(url, secs) {
  setBusy(false);
  const v = $("resultVideo"); v.src = url;
  view.classList.add("hidden"); v.classList.remove("hidden");
  $("downloadBtn").href = url;
  $("downloadBtn").download = (S.file.name.replace(/\.[^.]+$/, "") || "video") + "-clean.mp4";
  $("resultActions").classList.remove("hidden");
  $("hint").textContent = `Done in ${secs}s.`;
}
function showEditor() {
  view.classList.remove("hidden");
  $("compare").classList.add("hidden");
  $("resultVideo").classList.add("hidden"); $("resultVideo").removeAttribute("src");
  $("resultActions").classList.add("hidden");
  $("hint").textContent = S.mode === "image"
    ? "Paint over the watermark. Tip: draw a rough box around text, then Auto-detect to tighten the mask to the letters."
    : "Paint over the watermark on this frame — the mask applies to every frame. Auto-detect finds static logos for you.";
}
$("editAgainBtn").addEventListener("click", showEditor);
$("compareSlider").addEventListener("input", (e) => setCompare(+e.target.value));
function setCompare(v) {
  $("compareSlider").value = v;
  $("afterClip").style.clipPath = `inset(0 0 0 ${v}%)`;
  $("compare").style.setProperty("--pos", v + "%");
}

function setBusy(b) {
  $("runBtn").disabled = b || !hasMask();
  $("progress").classList.toggle("hidden", !b);
  if (!b) progress(0, "");
}
function progress(p, text) { $("bar").style.width = Math.round(p * 100) + "%"; $("progressText").textContent = text; }
function showError(m) { $("error").textContent = m; $("error").classList.toggle("hidden", !m); }

function reset() {
  clearTimeout(S.pollTimer);
  S.objectURLs.forEach(URL.revokeObjectURL); S.objectURLs = [];
  S.file = null; S.base = null; S.undo = [];
  $("fileInput").value = "";
  $("editor").classList.add("hidden"); drop.classList.remove("hidden");
  setBusy(false); showError(""); updateRun();
}

setMode("image");
