"""
segment.py
----------
High-accuracy GPU matting — replaces rembg-on-CPU.

Uses RMBG-2.0 (Bria, BiRefNet-based, current SOTA) on the L4 GPU to produce a
crisp, full-resolution CONTINUOUS ALPHA matte (not a downscaled binary mask).
This fixes inconsistent "somewhere sharp / somewhere smooth" edges at the source
and is far faster than CPU rembg.

Set MATTING_MODEL=ZhengPeng7/BiRefNet_HR for the open high-res (2048) alternative.
"""

import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

MODEL_ID = os.environ.get("MATTING_MODEL", "ZhengPeng7/BiRefNet_HR")   # open, MIT, high-res
RES = int(os.environ.get("MATTING_RES", "2048"))
device = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL = None
_TF = transforms.Compose([
    transforms.Resize((RES, RES)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def warmup():
    global _MODEL
    if _MODEL is None:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        m = AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True)
        m.to(device).eval()
        _MODEL = m
    return _MODEL


@torch.no_grad()
def cutout(image_pil):
    """RGB image -> RGBA with a crisp full-resolution alpha matte."""
    m = warmup()
    img = image_pil.convert("RGB")
    W, H = img.size
    mdtype = next(m.parameters()).dtype                    # match model dtype (fp16/fp32)
    x = _TF(img).unsqueeze(0).to(device=device, dtype=mdtype)
    preds = m(x)[-1].sigmoid().float().cpu()[0, 0]         # (RES, RES) in [0,1]
    mask = transforms.ToPILImage()(preds).resize((W, H), Image.LANCZOS)
    out = img.convert("RGBA")
    out.putalpha(mask)
    return out
