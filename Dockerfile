# ── V3 production image: FastAPI (:8000) + Gradio UI (/ui) ──
# CPU-only torch keeps the image usable on EC2 free-tier class hosts.
# Build from Shipment(V3)/ :  docker build -t identity-rag:v3 .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=0

WORKDIR /app

# system libs: git (CLIP from git), build tools (tokenizers/sentencepiece),
# libgomp (torch/scipy), tesseract (OCR fallback in loader.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgomp1 tesseract-ocr tesseract-ocr-eng curl \
 && rm -rf /var/lib/apt/lists/*

# torch CPU first (own layer — biggest, cached unless version changes)
RUN pip install --no-cache-dir \
    torch==2.12.0+cpu torchvision==0.27.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# remaining deps
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# app code (data/ + chroma mounted as volumes at runtime, not baked in)
COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/

# non-root runtime user; owns the mounted volumes
RUN useradd -m -u 1000 appuser \
 && mkdir -p /app/data /app/chroma_db /app/saved_index \
 && chown -R appuser:appuser /app
USER appuser

# ChromaDB + index resolve here at runtime (compose mounts volumes on top)
ENV CHROMA_PATH_OVERRIDE=/app/chroma_db \
    INDEX_PATH=/app/saved_index \
    DATA_DIR=/app/data

EXPOSE 8000

# single process: FastAPI serves the API + mounts Gradio at /ui
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
