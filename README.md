# AutoBG Demo — Car Background Replacement Studio

An AutoBG.ai-style demo: **upload a car → pick a studio background → the app
segments the car (object unchanged) and composites it onto the background with a
realistic shadow, reflection, and clean edges.**

Built for the Logics Yard "AI Image Generation & Background Enhancement" role.

---

## Why there is no GPU (read this — it's a key talking point)

This demo replaces the background with a **pre-made studio template**, so it is
**compositing, not AI generation**:

| Step | Compute |
|------|---------|
| Background removal (rembg / U²-Net) | **CPU**, ~2–5 s/image |
| Compositing (paste, shadow, reflection — OpenCV/Pillow) | **CPU**, milliseconds |
| Diffusion background *generation* (SDXL/Flux) | ❌ not used here |

A GPU (a 16 GB **T4**) would only be needed if we swapped templates for
**AI-generated** backgrounds via SDXL inpainting. For templates, CPU is enough —
so the whole thing runs for **~$0**.

---

## Architecture

```
Vercel (Next.js UI)  --HTTPS: image + template-->  GCP Cloud Run (FastAPI)
                                                     rembg     -> mask / cutout
                                                     OpenCV/PIL -> composite
                                                       + contact shadow
                                                       + floor reflection
                                                       + edge feather
                                               <-- final PNG
```

- **Frontend:** Next.js on **Vercel** (Vercel can't run the Python model — no GPU,
  function size/time limits — so it only hosts the UI).
- **Backend:** Dockerized **FastAPI** on **GCP Cloud Run** (serverless, scales to
  zero, CPU only).

The pipeline maps to prep stages 1–5: **segment → mask/alpha → edge-refine →
composite → shadow/reflection/light.**

---

## Run locally

### 1. Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate     |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```
Open http://localhost:8080/docs to test `POST /process` (upload a car + a template
id like `white_studio`). First call downloads the rembg model (~once).

### 2. Frontend
```bash
cd frontend
npm install
cp .env.local.example .env.local      # NEXT_PUBLIC_BACKEND_URL=http://localhost:8080
npm run dev
```
Open http://localhost:3000, upload a car, pick a background, click **Generate**.

---

## Deploy

### Backend → GCP Cloud Run
```bash
cd backend
gcloud run deploy autobg-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --set-env-vars ALLOWED_ORIGINS=https://YOUR-APP.vercel.app
```
Cloud Run builds the Docker image (via Cloud Build) and returns a public URL.
Copy that URL.

### Frontend → Vercel
1. Push `frontend/` to a Git repo and **Import** it in Vercel.
2. Set env var **`NEXT_PUBLIC_BACKEND_URL`** = the Cloud Run URL.
3. Deploy. Update the backend's `ALLOWED_ORIGINS` to the final Vercel domain.

---

## Project layout
```
backend/
  main.py            FastAPI app (/templates, /process, /cutout)
  compositing.py     templates + shadow + reflection + edge feather + composite
  requirements.txt
  Dockerfile         Cloud Run container (pre-pulls the rembg model)
frontend/
  app/page.js        upload UI, template picker, result + download
  app/globals.css    styling
  app/layout.js
```

---

## Easy upgrades (good things to mention in the interview)
- Swap **rembg → SAM** for higher-precision masks (promptable / YOLO-driven box).
- Add an **"AI background" mode**: SDXL-inpainting + ControlNet on a GPU worker
  for generated scenes instead of fixed templates.
- Move heavy/batch jobs onto an **async queue** (Cloud Tasks / Pub/Sub) with a
  job-id + polling or WebSocket progress.
