"""
compositing.py
--------------
Core image-processing logic for the AutoBG-style demo (CPU, no diffusion).

v4 — robust grounding that adapts to ANY car image / angle:
  segment (main.py) -> feather edges -> place
  -> detect TRUE contact line + footprint from the mask
  -> reflection anchored at contact (zero gap)
  -> two-part contact shadow (soft ambient + tight dark core) under the footprint
  -> composite.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageFilter

CANVAS_W, CANVAS_H = 1200, 900
PLATE_DIR = os.path.join(os.path.dirname(__file__), "plates")

TEMPLATES = {
    "white_studio": {"label": "White Studio", "wall_top": (239, 239, 240),
        "wall_bottom": (246, 246, 247), "floor_far": (244, 244, 246), "floor_near": (228, 228, 231),
        "horizon": 0.58, "glossy": True, "tiled": True, "baseboard": (208, 195, 170)},
    "dark_studio": {"label": "Dark Studio", "wall_top": (30, 32, 38),
        "floor_near": (40, 42, 50), "horizon": 0.64, "glossy": True, "glow": 34},
    "gradient_showroom": {"label": "Showroom Gradient", "wall_top": (196, 214, 234),
        "floor_near": (120, 150, 188), "horizon": 0.66, "glossy": True, "glow": 24},
}


def list_templates():
    return [{"id": k, "label": v["label"]} for k, v in TEMPLATES.items()]


def _vertical_gradient(height, width, color_top, color_bottom):
    top = np.array(color_top, dtype=np.float32)
    bottom = np.array(color_bottom, dtype=np.float32)
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    rows = top[None, :] * (1.0 - t) + bottom[None, :] * t
    return np.repeat(rows[:, None, :], width, axis=1).astype(np.uint8)


def _draw_floor_tiles(img, horizon_y, grout=(206, 206, 211)):
    """Draw a one-point-perspective tile grid on the floor region (in place).
    Larger tiles to match the reference studio."""
    H, W = img.shape[:2]
    floor_h = H - horizon_y
    vx = W // 2                                   # vanishing point x (centre)
    # horizontal tile lines — fewer = bigger tiles; bunch up near the horizon
    n = 8
    for i in range(1, n + 1):
        t = i / n
        yy = int(horizon_y + floor_h * (t ** 1.95))
        if horizon_y < yy < H:
            cv2.line(img, (0, yy), (W, yy), grout, 1, cv2.LINE_AA)
    # vertical tile lines — fewer, converge toward the vanishing point
    cols = 6
    spacing = W // cols
    for j in range(-cols - 3, cols + 4):
        xb = vx + j * spacing
        cv2.line(img, (xb, H - 1), (vx, horizon_y), grout, 1, cv2.LINE_AA)


def build_template(name):
    """Studio plate: a saved PNG plate if present, else a built wall+tile floor."""
    cfg = TEMPLATES.get(name) or TEMPLATES["white_studio"]
    horizon_y = int(CANVAS_H * cfg["horizon"])

    # Prefer a real background plate (e.g. the interviewer's studio with the car
    # removed) saved as plates/<name>.png — gives a pixel-exact, consistent background.
    plate = os.path.join(PLATE_DIR, name + ".png")
    if os.path.exists(plate):
        bg = Image.open(plate).convert("RGB").resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        return bg.convert("RGBA"), int(CANVAS_H * cfg.get("plate_horizon", cfg["horizon"])), cfg["glossy"]

    img = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    # wall (top) and floor (bottom), split at the horizon
    img[:horizon_y] = _vertical_gradient(
        horizon_y, CANVAS_W, cfg["wall_top"], cfg.get("wall_bottom", cfg["wall_top"]))
    img[horizon_y:] = _vertical_gradient(
        CANVAS_H - horizon_y, CANVAS_W,
        cfg.get("floor_far", cfg.get("floor_near")), cfg["floor_near"])

    # thin wood/tan baseboard where the wall meets the floor (like the reference)
    bb = cfg.get("baseboard")
    if bb:
        bb_h = max(5, int(CANVAS_H * 0.011))
        img[max(0, horizon_y - bb_h):horizon_y] = np.array(bb, dtype=np.float32)

    # faint contact line just below the baseboard
    img[horizon_y:horizon_y + 2] *= 0.95

    if cfg.get("tiled"):
        _draw_floor_tiles(img, horizon_y)

    # soft airy glow toward the centre
    yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W]
    cx, cy = CANVAS_W / 2.0, CANVAS_H * 0.42
    d = np.sqrt(((xx - cx) / (CANVAS_W * 0.8)) ** 2 + ((yy - cy) / (CANVAS_H * 0.72)) ** 2)
    glow = np.clip(1.0 - d, 0.0, 1.0)[:, :, None]
    img = (img + glow * cfg.get("glow", 8)).clip(0, 255)

    return Image.fromarray(img.astype(np.uint8), mode="RGB").convert("RGBA"), horizon_y, cfg["glossy"]


def _tight_bbox(alpha):
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return 0, 0, alpha.shape[1], alpha.shape[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def feather_edges(car_rgba):
    """Sharpen the matte (kills BiRefNet's semi-transparent ghost halo), trim
    the background rim, then anti-alias."""
    arr = np.array(car_rgba)
    alpha = arr[:, :, 3].astype(np.float32)
    alpha = np.clip((alpha - 50.0) * 2.2, 0, 255)
    a8 = alpha.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a8 = cv2.erode(a8, kernel, iterations=1)
    a8 = cv2.GaussianBlur(a8, (3, 3), 0)
    arr[:, :, 3] = a8
    return Image.fromarray(arr, mode="RGBA")


def autoscale_and_place(car_rgba, floor_line_y, width_ratio=0.66, max_h_ratio=0.46):
    arr = np.array(car_rgba)
    l, t, r, b = _tight_bbox(arr[:, :, 3])
    car = car_rgba.crop((l, t, r, b))
    target_w = CANVAS_W * width_ratio
    scale = target_w / car.width
    if car.height * scale > CANVAS_H * max_h_ratio:
        scale = (CANVAS_H * max_h_ratio) / car.height
    new_w = max(1, int(car.width * scale))
    new_h = max(1, int(car.height * scale))
    car = car.resize((new_w, new_h), Image.LANCZOS)
    x = (CANVAS_W - new_w) // 2
    # place so the car's bottom lands on the floor line
    y = floor_line_y - new_h
    return car, x, y


def contact_info(car_img):
    """
    Detect ground-contact line + footprint from the alpha mask (not the bbox).
    Adapts to any car shape/angle:
      contact_y = lowest opaque row (tires-on-ground),
      fx0, fx1  = car's horizontal extent in the bottom contact band.
    """
    a = np.array(car_img)[:, :, 3]
    mask = a > 40
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return car_img.height - 1, 0, car_img.width - 1
    contact_y = int(rows.max())
    band_top = max(0, contact_y - int(car_img.height * 0.12))
    band = mask[band_top:contact_y + 1, :]
    bcols = np.where(band.any(axis=0))[0]
    if len(bcols) == 0:
        cols = np.where(mask.any(axis=0))[0]
        return contact_y, int(cols.min()), int(cols.max())
    return contact_y, int(bcols.min()), int(bcols.max())


def make_ground_shadow(car_img, x, y, contact_y, fx0, fx1):
    """
    Anchored contact shadow under the real footprint:
      - soft ambient ellipse (diffuse light),
      - tight dark core ellipse right at the wheels (kills 'floating').
    Footprint-driven, so it adapts to the car's width and angle.
    """
    ch = car_img.height
    gy = y + contact_y                       # absolute ground line
    foot_cx = x + (fx0 + fx1) // 2
    foot_w = max(14, fx1 - fx0)

    layer = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

    # 1) soft ambient
    cv2.ellipse(layer, (foot_cx, gy),
                (int(foot_w * 0.60), max(7, int(ch * 0.10))), 0, 0, 360, 0.42, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=24, sigmaY=10)

    # 2) tight dark core (the anchor)
    core = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    cv2.ellipse(core, (foot_cx, gy),
                (int(foot_w * 0.50), max(4, int(ch * 0.035))), 0, 0, 360, 1.0, -1)
    core = cv2.GaussianBlur(core, (0, 0), sigmaX=11, sigmaY=4)

    combined = np.clip(layer + core * 0.65, 0, 1)
    a8 = (combined * 255).astype(np.uint8)
    patch = np.zeros((CANVAS_H, CANVAS_W, 4), dtype=np.uint8)
    patch[:, :, 3] = a8
    return Image.fromarray(patch, mode="RGBA")


def make_reflection(car_img, x, y, contact_y, opacity=0.16):
    """Mirror the car about the CONTACT line (zero gap) + fade + slight blur."""
    car_top = car_img.crop((0, 0, car_img.width, max(1, contact_y)))
    refl = car_top.transpose(Image.FLIP_TOP_BOTTOM)
    arr = np.array(refl).astype(np.float32)
    h = arr.shape[0]
    grad = (np.linspace(1.0, 0.0, h, dtype=np.float32) ** 1.8)[:, None]
    arr[:, :, 3] = arr[:, :, 3] * grad * opacity
    refl = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGBA")
    refl = refl.filter(ImageFilter.GaussianBlur(1.2))
    out = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    out.alpha_composite(refl, (x, y + contact_y))
    return out


def composite(car_rgba, template_name):
    bg, floor_line_y, glossy = build_template(template_name)
    car = feather_edges(car_rgba)
    # sit the car DOWN on the floor (in front of the wall), not at the wall/floor line
    car_ground = int(CANVAS_H * 0.74)
    car, x, y = autoscale_and_place(car, car_ground)
    contact_y, fx0, fx1 = contact_info(car)
    if glossy:
        bg.alpha_composite(make_reflection(car, x, y, contact_y))
    bg.alpha_composite(make_ground_shadow(car, x, y, contact_y, fx0, fx1))
    bg.alpha_composite(car, (x, y))
    return bg.convert("RGB")
