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

try:
    from pymatting import estimate_foreground_ml
    _HAS_PYMATTING = True
except Exception:
    _HAS_PYMATTING = False

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


def _decontaminate(car_rgba):
    """Foreground colour decontamination (the 'Defringe' / 'Decontaminate Colors' fix):
    grow the car's trusted interior colours OUTWARD into the semi-transparent edge band,
    so the rim carries CAR colour instead of the original photo's background bleed. This
    is what kills the white halo when compositing onto a different (studio) background."""
    arr = np.array(car_rgba)
    rgb = arr[:, :, :3].astype(np.float32)
    a = arr[:, :, 3].astype(np.float32) / 255.0
    solid = (a > 0.75).astype(np.float32)              # colours we trust (opaque interior)
    if solid.sum() == 0:
        return car_rgba
    col = rgb * solid[:, :, None]                      # premultiplied known colours
    w = solid.copy()
    for _ in range(8):                                 # diffuse interior colour outward
        col = cv2.GaussianBlur(col, (0, 0), 2.0)
        w = cv2.GaussianBlur(w, (0, 0), 2.0)
        col = col * (1 - solid)[:, :, None] + rgb * solid[:, :, None]   # re-assert interior
        w = w * (1 - solid) + solid
    grown = col / (w[:, :, None] + 1e-6)
    edge = np.clip((0.95 - a) / 0.95, 0, 1)[:, :, None]                 # 0 interior -> 1 rim
    rgb2 = rgb * (1 - edge) + grown * edge                              # rim takes car colour
    arr[:, :, :3] = np.clip(rgb2, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _estimate_fg(rgb01, a01, cap=1280):
    """True foreground colour via PyMatting unmixing (I = aF + (1-a)B  ->  solve F).
    Capped resolution for speed: F is smooth, so estimate at <=cap px and upscale."""
    h, w = a01.shape
    s = min(1.0, cap / float(max(h, w)))
    if s < 1.0:
        rs = (max(1, int(w * s)), max(1, int(h * s)))
        img_s = cv2.resize(rgb01, rs, interpolation=cv2.INTER_AREA)
        a_s = cv2.resize(a01, rs, interpolation=cv2.INTER_AREA)
        F = cv2.resize(estimate_foreground_ml(img_s, a_s), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        F = estimate_foreground_ml(rgb01, a01)
    return np.clip(F, 0.0, 1.0)


def feather_edges(car_rgba):
    """Production edge pipeline (accurate + efficient):
      1. FOREGROUND ESTIMATION — unmix the true car colour (kills the white halo at its
         source; FBA/PyMatting). No background bleed survives onto the studio.
      2. Gentle matte refine — contract 1px to drop any background over-extension, then
         anti-alias. No aggressive binarisation (which caused aliasing + ate thin parts)."""
    arr = np.array(car_rgba)
    a01 = arr[:, :, 3].astype(np.float64) / 255.0
    if a01.max() == 0:
        return car_rgba

    if _HAS_PYMATTING:
        rgb01 = arr[:, :, :3].astype(np.float64) / 255.0
        F = _estimate_fg(rgb01, a01)
        rgb_out = (F * 255.0).clip(0, 255).astype(np.uint8)
    else:                                              # safe fallback if pkg missing
        rgb_out = np.array(_decontaminate(car_rgba))[:, :, :3]

    a8 = arr[:, :, 3]
    a8 = cv2.erode(a8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    a8 = cv2.GaussianBlur(a8, (3, 3), 0)
    return Image.fromarray(np.dstack([rgb_out, a8]), mode="RGBA")


def autoscale_and_place(car_rgba, floor_line_y, width_ratio=0.70, max_h_ratio=0.56):
    """Real-world size normalization: scale every car to a consistent on-floor LENGTH
    (width target). The height cap is relaxed so tall vehicles (SUVs) keep that length
    and simply stand taller — instead of being shrunk to fit — giving consistent scale
    across sedans / supercars / SUVs."""
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
    Contour-following contact shadow: lays the shadow along the car's actual bottom
    silhouette (lowest opaque pixel per column), so it grounds the FRONT wheel (low
    in a 3/4 view) AND the REAR wheel (higher) — no more 'lifted at the back'.
    Soft ambient band + tighter dark core, tinted dark-grey (not pure black).
    """
    a = np.array(car_img)[:, :, 3]
    nh, nw = a.shape
    mask = a > 40
    if mask.sum() == 0:
        return Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    # bottom contour: lowest opaque row per column -> map to canvas ground y per column
    idx = np.where(mask, np.arange(nh)[:, None], -1)
    lowest = idx.max(axis=0)
    ground = np.full(CANVAS_W, -1, dtype=np.int32)
    cols = np.where(lowest >= 0)[0]
    gx = np.clip(x + cols, 0, CANVAS_W - 1)
    gy = y + lowest[cols]
    inb = (gy >= 0) & (gy < CANVAS_H)
    ground[gx[inb]] = gy[inb]

    rows = np.arange(CANVAS_H)[:, None]
    g = ground[None, :]
    vmask = g >= 0
    drop = max(8, int(nh * 0.05))

    # soft ambient band hugging the contour
    band = ((rows >= g - 3) & (rows <= g + drop) & vmask).astype(np.float32)
    shadow = cv2.GaussianBlur(band, (0, 0), sigmaX=16, sigmaY=8) * 0.5

    # tighter dark core right at the contact line (grounds the wheels)
    core_band = ((rows >= g - 1) & (rows <= g + max(3, int(nh * 0.02))) & vmask).astype(np.float32)
    core = cv2.GaussianBlur(core_band, (0, 0), sigmaX=6, sigmaY=2)
    shadow = np.clip(shadow + core * 0.55, 0, 1)

    patch = np.zeros((CANVAS_H, CANVAS_W, 4), dtype=np.uint8)
    patch[:, :, 0] = 30; patch[:, :, 1] = 30; patch[:, :, 2] = 34   # tinted, not pure black
    patch[:, :, 3] = (shadow * 255).astype(np.uint8)
    return Image.fromarray(patch, mode="RGBA")


def make_reflection(car_img, x, y, contact_y, opacity=0.40, compress=0.72):
    """
    Perspective-aware floor reflection: each column is mirrored about ITS OWN
    contact point (seamless on 3/4 views), vertically COMPRESSED (foreshortened,
    since the floor recedes from the camera), then faded + blurred with distance.
    Far more realistic than a flat full-car flip.
    """
    arr = np.array(car_img).astype(np.float32)
    nh, nw, _ = arr.shape
    mask = arr[:, :, 3] > 40
    idx = np.where(mask, np.arange(nh)[:, None], -1)
    contour = idx.max(axis=0)                      # per-column lowest opaque row
    refl = np.zeros((CANVAS_H, CANVAS_W, 4), dtype=np.float32)
    rows = np.arange(CANVAS_H)
    fade_len = max(2.0, nh * 0.45 * compress)      # canvas rows over which it fades out

    for c in range(nw):
        Lc = contour[c]
        if Lc < 0:
            continue
        gx = x + c
        if gx < 0 or gx >= CANVAS_W:
            continue
        base = y + Lc                              # this column's contact row on canvas
        db = rows - base                           # distance below the contact
        v = db >= 0
        src = (Lc - db / compress)                 # walk UP the car (de-compressed)
        v &= (src >= 0) & (src <= Lc)
        rr = rows[v]
        sr = src[v].astype(np.int32)
        fade = np.clip(1.0 - db[v] / fade_len, 0, 1) ** 1.6
        refl[rr, gx, 0:3] = arr[sr, c, 0:3]
        refl[rr, gx, 3] = arr[sr, c, 3] * fade * opacity

    out = Image.fromarray(refl.clip(0, 255).astype(np.uint8), mode="RGBA")
    out = out.filter(ImageFilter.GaussianBlur(1.5))
    return out


# ---- Phase 2: deterministic realism pass (no models, no new deps) -------------
REALISM = os.environ.get("REALISM", "1") != "0"
WB_STRENGTH = float(os.environ.get("WB_STRENGTH", "0.15"))   # studio white-balance
LIGHTWRAP = float(os.environ.get("LIGHTWRAP", "0.12"))       # bg light onto edges
GRAIN = float(os.environ.get("GRAIN", "2.5"))                # unify fg/bg texture
BOUNCE = float(os.environ.get("BOUNCE", "0.10"))             # ambient floor-bounce relight


def _harmonize_wb(car, strength=WB_STRENGTH):
    """Neutralize the car's colour cast toward the studio's neutral white light
    (shift a,b channel means toward 128). Touches ONLY white balance — never the
    car's exposure/contrast/geometry — so it can't wash out or distort the car."""
    arr = np.array(car)
    m = arr[:, :, 3] > 10
    if m.sum() == 0:
        return car
    lab = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    for i in (1, 2):                                        # a, b -> toward neutral 128
        ch = lab[:, :, i]
        lab[:, :, i] = ch + (128.0 - ch[m].mean()) * strength
    arr[:, :, :3] = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(arr, "RGBA")


def _ambient_bounce(car, lift=BOUNCE, floor_bias=0.55):
    """'Normalize lighting to match the backdrop' (what Spyne/CarStudio do): lift the
    car's SHADOW regions toward the bright studio, weighted toward the lower body
    (white-floor bounce). Only fills shadows — highlights & geometry untouched — so the
    car reads as lit by this room without being washed out or distorted."""
    arr = np.array(car)
    m = arr[:, :, 3] > 10
    if m.sum() == 0:
        return car
    lab = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    nh = arr.shape[0]
    grad = np.linspace(1.0 - floor_bias, 1.0, nh, dtype=np.float32)[:, None]   # bottom lifts more
    shadow_w = np.clip(1.0 - L / 255.0, 0, 1)                                  # darker = lift more
    addL = lift * 255.0 * grad * shadow_w
    lab[:, :, 0] = np.where(m, np.clip(L + addL, 0, 255), L)
    arr[:, :, :3] = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(arr, "RGBA")


def _light_wrap(comp, scene_bg, car, x, y, width=14, blur=18, strength=LIGHTWRAP):
    """Wrap the studio light onto the car's edges (screen-blend a blurred copy of the
    background into a thin band just inside the silhouette) so the edge stops reading
    as 'pasted', while the car stays crisp."""
    W, H = comp.size
    ca = np.array(car)[:, :, 3]
    nh, nw = ca.shape
    A = np.zeros((H, W), np.float32)
    A[y:y + nh, x:x + nw] = ca / 255.0
    Ab = (A > 0.5).astype(np.uint8)
    er = cv2.erode(Ab, np.ones((width, width), np.uint8))
    rim = ((Ab - er) > 0).astype(np.float32)
    rim = cv2.GaussianBlur(rim, (0, 0), width / 2.0) * A
    c = np.array(comp).astype(np.float32) / 255.0
    b = cv2.GaussianBlur(np.array(scene_bg).astype(np.float32) / 255.0, (0, 0), blur)
    screen = 1.0 - (1.0 - c) * (1.0 - b)
    wr = (rim * strength)[:, :, None]
    out = c * (1.0 - wr) + screen * wr
    return Image.fromarray((out * 255).clip(0, 255).astype(np.uint8), "RGB")


def _add_grain(img, amount=GRAIN):
    """Subtle, seeded (deterministic) mono film grain to unify fg/bg texture."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.default_rng(0).normal(0.0, amount, arr.shape[:2])[:, :, None]
    return Image.fromarray((arr + noise).clip(0, 255).astype(np.uint8), "RGB")


def composite(car_rgba, template_name):
    bg, floor_line_y, glossy = build_template(template_name)
    scene_bg = bg.copy()                                   # clean studio (light-wrap ref)
    car = feather_edges(car_rgba)
    if REALISM:
        car = _harmonize_wb(car)                           # match studio white balance
        car = _ambient_bounce(car)                         # normalize lighting to backdrop
    # sit the car DOWN on the floor (in front of the wall), not at the wall/floor line
    car_ground = int(CANVAS_H * 0.74)
    car, x, y = autoscale_and_place(car, car_ground)
    contact_y, fx0, fx1 = contact_info(car)
    if glossy:
        bg.alpha_composite(make_reflection(car, x, y, contact_y))
    bg.alpha_composite(make_ground_shadow(car, x, y, contact_y, fx0, fx1))
    bg.alpha_composite(car, (x, y))
    out = bg.convert("RGB")
    if REALISM:
        out = _light_wrap(out, scene_bg.convert("RGB"), car, x, y)   # integrate edges
        out = _add_grain(out)                                        # unify texture
    return out
