"""
make_plate.py
-------------
Build a clean studio background PLATE by REMOVING the car from a reference photo.

Put the reference image at  plates/reference.<png|jpg>  and run:
    python make_plate.py
Output:  plates/white_studio.png  — the same studio, car removed (SDXL-inpainted).

The compositing pipeline then pastes any car onto this exact background.
"""

import os
import glob
import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter

import segment
import aigen

HERE = os.path.dirname(os.path.abspath(__file__))
PLATES = os.path.join(HERE, "plates")
OUT = os.path.join(PLATES, "white_studio.png")
SIZE = 1024


def main():
    refs = sorted(glob.glob(os.path.join(PLATES, "reference.*")))
    if not refs:
        raise SystemExit("Put the reference image at plates/reference.png (or .jpg) first.")
    ref = Image.open(refs[0]).convert("RGB")
    W, H = ref.size
    print("reference:", refs[0], ref.size)

    # 1) segment the car -> binary mask, dilated to also cover its shadow/reflection
    cut = segment.cutout(ref)
    alpha = np.array(cut)[:, :, 3]
    mask = (alpha > 40).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((35, 35), np.uint8))

    # 2) SDXL-inpaint the masked region with an empty studio (reuse aigen's pipeline;
    #    ControlNet disabled so it freely reconstructs the wall + tiled floor)
    pipe = aigen._load()
    ref_s = ref.resize((SIZE, SIZE), Image.LANCZOS)
    mask_s = Image.fromarray(mask).convert("L").resize((SIZE, SIZE)).filter(ImageFilter.GaussianBlur(6))
    blank = Image.fromarray(np.zeros((SIZE, SIZE, 3), np.uint8))
    gen = torch.Generator(device="cpu").manual_seed(3)
    out = pipe(
        prompt=("empty bright white photo studio, seamless white wall, glossy white tiled floor "
                "with soft reflections, minimalist, clean, no car, no vehicle, photorealistic, ultra detailed"),
        negative_prompt="car, vehicle, suv, sedan, object, person, clutter, text, watermark",
        image=ref_s, mask_image=mask_s, control_image=blank,
        num_inference_steps=30, guidance_scale=7.0, strength=1.0,
        controlnet_conditioning_scale=0.0, height=SIZE, width=SIZE, generator=gen,
    ).images[0].resize((W, H), Image.LANCZOS)

    os.makedirs(PLATES, exist_ok=True)
    out.save(OUT)
    print("saved plate ->", OUT)


if __name__ == "__main__":
    main()
