FROM python:3.11-slim

# System libs required by OpenCV (headless) and onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the rembg model at build time so the container starts fast
# (avoids a cold first-request download on Cloud Run).
ENV REMBG_MODEL=isnet-general-use
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

COPY . .

# Cloud Run injects $PORT (default 8080). Bind to it.
ENV PORT=8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
