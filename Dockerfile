# ---- ClearFrame: watermark & object removal for images and video ----
FROM python:3.12-slim

ARG WITH_LAMA=1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LAMA_MODEL_PATH=/app/models/lama_fp32.onnx \
    WORK_DIR=/tmp PORT=8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# model layer before code so code changes don't re-download 200 MB
COPY scripts/download_model.sh scripts/
RUN if [ "$WITH_LAMA" = "1" ]; then bash scripts/download_model.sh "$LAMA_MODEL_PATH"; fi

COPY app app
COPY static static

RUN useradd -r -u 10001 app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD curl -fs http://localhost:${PORT}/api/health || exit 1
# one worker: in-memory job store + model RAM. Scale with replicas, not workers.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
