"""AutoBG backend on L4. GPU matting (RMBG-2.0) + template / ai / relight modes."""
import io, os
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import compositing
import segment

app = FastAPI(title="AutoBG L4 Backend", version="7.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MAX_SIDE = int(os.environ.get("MAX_INPUT_SIDE", "2048"))


def _downscale(img, m=MAX_SIDE):
    w, h = img.size
    if max(w, h) <= m:
        return img
    s = m / max(w, h)
    return img.resize((int(w * s), int(h * s)), Image.LANCZOS)


@app.on_event("startup")
def _load():
    segment.warmup()                       # load RMBG-2.0 onto the GPU


@app.get("/")
def health():
    return {"status": "ok", "matting": segment.MODEL_ID}


@app.get("/templates")
def tpls():
    return {"templates": compositing.list_templates()}


@app.post("/process")
async def process(file: UploadFile = File(...), template: str = Form("white_studio"),
                  mode: str = Form("template")):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")
    try:
        src = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image.")
    src = _downscale(src)

    try:
        cut = segment.cutout(src)          # crisp GPU matte (RGBA)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    try:
        if mode == "ai":
            import aigen
            result = aigen.generate(cut, template)
        else:
            result = compositing.composite(cut, template)
    except Exception as e:
        raise HTTPException(500, f"{mode} generation failed: {e}")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
