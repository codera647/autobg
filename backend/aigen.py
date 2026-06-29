"""
aigen.py
--------
"AI Background" mode v4 — SDXL ControlNet INPAINTING + crisp edges + grounded
wheel contact shadow.

Pipeline:
  segment -> place car on canvas -> background inpaint mask + canny
  -> SDXL ControlNet inpaint generates the premium scene
  -> draw a TWO-TIER contact shadow on the generated floor (soft ambient under
     the body + hard dark cores under each wheel; ambient-tinted, fades w/ dist)
  -> paste the car back with an EDGE-AWARE (guided-filter) crisp matte.

v4 fixes: blurry tyre edges (guided-filter refinement instead of uniform blur)
and floating wheels (dedicated hard contact shadow under the tyres).
"""

import os
import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
)

SIZE = int(os.environ.get("SDXL_SIZE", "1024"))
STEPS = int(os.environ.get("SDXL_STEPS", "30"))
GUIDANCE = float(os.environ.get("SDXL_GUIDANCE", "7.0"))
CN_SCALE = float(os.environ.get("CN_SCALE", "0.5"))
MASK_FEATHER = int(os.environ.get("MASK_FEATHER", "2"))      # less boundary blur
HARM_L = float(os.environ.get("HARMONIZE_L", "0.35"))        # exposure match strength
HARM_C = float(os.environ.get("HARMONIZE_C", "0.15"))        # colour/white-balance match strength
DEGLOW = float(os.environ.get("DEGLOW", "0.7"))              # floor-halo removal strength (0 = off)
LIGHTWRAP = float(os.environ.get("LIGHTWRAP", "0.5"))        # wrap bg light onto car edges (0 = off)
GRAIN = float(os.environ.get("GRAIN", "3.5"))                # film grain amount to unify fg/bg (0 = off)

BASE_ID = os.environ.get("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
CN_ID = os.environ.get("CN_MODEL", "diffusers/controlnet-canny-sdxl-1.0")
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

PROMPTS = {
    "white_studio": ("professional automotive advertising photograph, parked in a bright seamless white "
        "photo studio, soft diffused overhead studio lighting, glossy reflective white floor with a soft "
        "reflection, premium luxury car commercial, photorealistic, ultra detailed"),
    "dark_studio": ("professional automotive advertising photograph, in a dark luxury studio, dramatic soft "
        "rim lighting, glossy black reflective floor, moody premium showroom, cinematic lighting, photorealistic, ultra detailed"),
    "gradient_showroom": ("professional automotive advertising photograph, in a modern luxury car showroom "
        "with large bright windows, soft natural daylight, polished floor with reflection, premium dealership "
        "interior, photorealistic, ultra detailed"),
}
NEG = ("blurry, distorted, deformed, extra wheels, duplicate, cartoon, illustration, low quality, jpeg "
       "artifacts, text, watermark, logo, people, person, cluttered, messy, glowing outline, bright halo, "
       "rim light glow, light spill, white fringe, harsh edges, overexposed floor, blown highlights, "
       "bright glow around car, light bleed, halo on floor")

_PIPE = None


def _load():
    global _PIPE
    if _PIPE is None:
        cn = ControlNetModel.from_pretrained(CN_ID, torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            BASE_ID, controlnet=cn, vae=vae, torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        _PIPE = pipe
    return _PIPE


def _compose(car_rgba, size=SIZE):
    arr = np.array(car_rgba)
    ys, xs = np.where(arr[:, :, 3] > 10)
    if len(xs):
        car = car_rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    else:
        car = car_rgba
    tw = int(size * 0.80)
    sc = tw / car.width
    if car.height * sc > size * 0.60:
        sc = size * 0.60 / car.height
    nw, nh = max(1, int(car.width * sc)), max(1, int(car.height * sc))
    car = car.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    x = (size - nw) // 2
    y = max(0, int(size * 0.60) - nh)
    canvas.paste(car, (x, y), car)

    mask = Image.new("L", (size, size), 255)
    mask.paste(Image.new("L", (nw, nh), 0), (x, y), car.split()[3])
    if MASK_FEATHER > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(MASK_FEATHER))

    a = np.array(canvas)
    edges = cv2.Canny(a, 80, 180)
    control = Image.fromarray(np.stack([edges, edges, edges], axis=-1))
    return canvas, mask, control, car, (x, y)


# ----------------------------------------------------------------------------
# Part 1: edge-aware crisp matte (hand-rolled guided filter, no extra deps)
# ----------------------------------------------------------------------------
def _guided_filter(guide, src, radius=4, eps=1e-4):
    """Edge-preserving refinement of `src` (alpha) guided by `guide` (gray). 0..1 floats."""
    r = (radius, radius)
    mean_I = cv2.boxFilter(guide, -1, r)
    mean_p = cv2.boxFilter(src, -1, r)
    mean_Ip = cv2.boxFilter(guide * src, -1, r)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, -1, r)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, r) * guide + cv2.boxFilter(b, -1, r)


def _clean_edge(car):
    """Refine the alpha against the car image so tyre/rim edges stay CRISP."""
    arr = np.array(car).astype(np.float32)
    guide = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    alpha = arr[:, :, 3] / 255.0
    refined = _guided_filter(guide, alpha, radius=4, eps=1e-4)
    refined = np.clip((refined - 0.5) * 1.6 + 0.5, 0.0, 1.0)   # sharpen the transition
    arr[:, :, 3] = (refined * 255.0).clip(0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


# ----------------------------------------------------------------------------
# Part 2: two-tier wheel contact shadow on the generated scene
# ----------------------------------------------------------------------------
def _add_contact_shadow(scene, car, x, y):
    W, H = scene.size
    ca = np.array(car)[:, :, 3]
    nh, nw = ca.shape
    mask = ca > 40
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return scene

    # per-column lowest opaque row (local coords)
    idx = np.where(mask, np.arange(nh)[:, None], -1)
    lowest = idx.max(axis=0)
    contact_y_local = int(lowest.max())
    contact_y = y + contact_y_local
    fx0, fx1 = int(cols.min()), int(cols.max())
    foot_w = max(14, fx1 - fx0)
    fcx = x + (fx0 + fx1) // 2

    layer = np.zeros((H, W), np.float32)
    # soft ambient shadow under the whole body
    cv2.ellipse(layer, (fcx, contact_y), (int(foot_w * 0.58), max(8, int(nh * 0.07))),
                0, 0, 360, 0.30, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=28, sigmaY=12)

    # hard dark cores under the wheels (columns reaching near the contact line)
    cores = np.zeros((H, W), np.float32)
    thresh = contact_y_local - int(nh * 0.04)
    wheelcols = [int(c) for c in cols if lowest[c] >= thresh]
    if wheelcols:
        segs, s, p = [], wheelcols[0], wheelcols[0]
        for c in wheelcols[1:]:
            if c - p > 6:
                segs.append((s, p)); s = c
            p = c
        segs.append((s, p))
        for (a0, a1) in segs:
            if (a1 - a0) < nw * 0.02:
                continue
            wcx = x + (a0 + a1) // 2
            ww = a1 - a0
            cv2.ellipse(cores, (wcx, contact_y + int(nh * 0.01)),
                        (int(ww * 0.6), max(4, int(nh * 0.022))), 0, 0, 360, 0.7, -1)
        cores = cv2.GaussianBlur(cores, (0, 0), sigmaX=6, sigmaY=2)

    shadow = np.clip(layer + cores, 0.0, 1.0)[:, :, None]
    tint = np.array([34, 34, 40], np.float32)        # ambient-tinted, not pure black
    sc = np.array(scene).astype(np.float32)
    sc = sc * (1.0 - shadow) + tint[None, None, :] * shadow
    return Image.fromarray(sc.clip(0, 255).astype(np.uint8), "RGB")


def _harmonize(car, scene, l_strength=HARM_L, c_strength=HARM_C):
    """
    Deterministic illumination/colour harmonization (LAB statistics transfer):
    nudge the car's exposure (L) and white-balance (a,b) toward the generated
    scene so it sits in the studio light. Adjusts ONLY colour, never geometry —
    so it can't distort the car (unlike generative relighting).
    """
    arr = np.array(car)
    m = arr[:, :, 3] > 10
    if m.sum() == 0:
        return car
    car_lab = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    scene_lab = cv2.cvtColor(np.array(scene), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    s_mean, s_std = scene_lab.mean(0), scene_lab.std(0) + 1e-6
    strengths = (l_strength, c_strength, c_strength)
    for i in range(3):
        ch = car_lab[:, :, i]
        cm, cs = ch[m].mean(), ch[m].std() + 1e-6
        matched = (ch - cm) * (s_std[i] / cs) + s_mean[i]
        car_lab[:, :, i] = ch * (1 - strengths[i]) + matched * strengths[i]
    arr[:, :, :3] = cv2.cvtColor(car_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(arr, "RGBA")


def _deglow(scene, car, x, y, radius=70, l_thresh=10.0, strength=0.7):
    """
    Tone down the over-bright 'light spill' SDXL generates around the car — using
    a SMOOTH distance-weighted falloff (no hard ring => no boundary/box artifact).
    Only the EXCESS brightness above the floor tone is reduced, weighted by how
    close the pixel is to the car, fading smoothly to zero by `radius` px.
    """
    W, H = scene.size
    ca = np.array(car)[:, :, 3]
    nh, nw = ca.shape
    cm = np.zeros((H, W), np.uint8)
    cm[y:y + nh, x:x + nw] = ca
    car_mask = (cm > 40).astype(np.uint8)
    if car_mask.sum() == 0:
        return scene

    dist = cv2.distanceTransform(1 - car_mask, cv2.DIST_L2, 5)     # px distance from car
    w = np.clip(1.0 - dist / float(radius), 0, 1).astype(np.float32)
    w = cv2.GaussianBlur(w, (0, 0), 10) * (1 - car_mask)           # smooth, exclude car

    lab = cv2.cvtColor(np.array(scene), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    far = dist > radius
    ref = float(np.median(L[far])) if far.sum() > 0 else float(np.median(L))
    excess = np.clip(L - (ref + l_thresh), 0, None)               # only over-bright pixels
    L = L - excess * w * strength
    lab[:, :, 0] = L
    out = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out, "RGB")


def _light_wrap(comp, scene_bg, car, x, y, width=14, blur=18, strength=LIGHTWRAP):
    """
    Wrap the background light onto the car's edges (classic compositing trick):
    screen-blend a blurred copy of the background into a thin band just inside the
    car's silhouette. Kills the hard 'pasted' edge while keeping the car crisp.
    """
    W, H = comp.size
    ca = np.array(car)[:, :, 3]
    nh, nw = ca.shape
    A = np.zeros((H, W), np.float32)
    A[y:y + nh, x:x + nw] = ca / 255.0
    Ab = (A > 0.5).astype(np.uint8)
    er = cv2.erode(Ab, np.ones((width, width), np.uint8))
    rim = ((Ab - er) > 0).astype(np.float32)
    rim = cv2.GaussianBlur(rim, (0, 0), width / 2.0) * A          # soft band inside the edge
    c = np.array(comp).astype(np.float32) / 255.0
    b = cv2.GaussianBlur(np.array(scene_bg).astype(np.float32) / 255.0, (0, 0), blur)
    screen = 1.0 - (1.0 - c) * (1.0 - b)                          # background light
    wr = (rim * strength)[:, :, None]
    out = c * (1.0 - wr) + screen * wr
    return Image.fromarray((out * 255).clip(0, 255).astype(np.uint8), "RGB")


def _add_grain(img, amount=GRAIN):
    """Uniform mono film grain over the whole composite to unify fg/bg texture."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0.0, amount, arr.shape[:2])[:, :, None]
    return Image.fromarray((arr + noise).clip(0, 255).astype(np.uint8), "RGB")


def generate(car_rgba, template_key):
    pipe = _load()
    prompt = PROMPTS.get(template_key, PROMPTS["white_studio"])
    canvas, mask, control, car, (x, y) = _compose(car_rgba)

    gen = torch.Generator(device="cpu").manual_seed(7)
    out = pipe(
        prompt=prompt, negative_prompt=NEG,
        image=canvas, mask_image=mask, control_image=control,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE, strength=1.0,
        controlnet_conditioning_scale=CN_SCALE, height=SIZE, width=SIZE, generator=gen,
    ).images[0].convert("RGB")

    if DEGLOW > 0:
        out = _deglow(out, car, x, y, strength=DEGLOW)   # smooth floor-halo removal
    out = _add_contact_shadow(out, car, x, y)      # ground the wheels
    car = _harmonize(car, out)                      # match car illumination to scene
    cc = _clean_edge(car)                           # crisp matte
    bg_for_wrap = out.copy()                         # background before the car
    out.paste(cc, (x, y), cc)
    if LIGHTWRAP > 0:
        out = _light_wrap(out, bg_for_wrap, cc, x, y)   # integrate edges (anti-pasted)
    if GRAIN > 0:
        out = _add_grain(out)                            # unify fg/bg texture
    return out
