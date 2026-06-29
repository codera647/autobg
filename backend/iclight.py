"""
iclight.py
----------
IC-Light FBC (foreground + background conditioned) relighting — SEPARATE mode.

Relights the car to match a studio background's lighting. Based on
lllyasviel/IC-Light: SD1.5 UNet patched conv_in 4->12 channels, iclight_sd15_fbc
offset weights merged additively, and a hooked UNet forward that concatenates the
VAE-encoded foreground + background latents to the noisy latent each step.

Runs at 512 (SD1.5) and upscales to 1024. Lazy-loaded; independent of aigen.

Env: ICL_STEPS (25), ICL_GUIDANCE (7.0), ICL_BASE.
"""

import os
import numpy as np
import torch
from PIL import Image
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline, DDIMScheduler

import compositing  # for the studio background image

device = "cuda"
dtype = torch.float16
STEPS = int(os.environ.get("ICL_STEPS", "25"))
GUID = float(os.environ.get("ICL_GUIDANCE", "7.0"))
BASE = os.environ.get("ICL_BASE", "stablediffusionapi/realistic-vision-v51")

PROMPTS = {
    "white_studio": "car in a bright white photo studio, soft even studio lighting, professional automotive product photography",
    "dark_studio": "car in a dark luxury studio, dramatic soft studio lighting, professional automotive product photography",
    "gradient_showroom": "car in a modern luxury showroom, soft natural daylight, professional automotive product photography",
}
NEG = "lowres, bad quality, distorted, deformed, extra wheels, watermark, text, cartoon"

_PIPE = None
_VAE = None


def _load():
    global _PIPE, _VAE
    if _PIPE is not None:
        return _PIPE
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    unet = pipe.unet

    # patch conv_in: 4 -> 12 channels (first 4 = original weights, rest zero)
    with torch.no_grad():
        nc = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size,
                             unet.conv_in.stride, unet.conv_in.padding)
        nc.weight.zero_()
        nc.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        nc.bias.copy_(unet.conv_in.bias)
        unet.conv_in = nc

    # merge IC-Light FBC offset weights additively
    path = hf_hub_download("lllyasviel/ic-light", "iclight_sd15_fbc.safetensors")
    offset = sf.load_file(path)
    sd = unet.state_dict()
    merged = {k: ((sd[k].float() + offset[k].float()).to(dtype) if k in offset else sd[k])
              for k in sd}
    unet.load_state_dict(merged)
    unet = unet.to(device, dtype)

    # hook the forward to concatenate the fg+bg conditioning latents
    orig_forward = unet.forward

    def hooked(sample, timestep, encoder_hidden_states, **kw):
        cak = dict(kw.get("cross_attention_kwargs") or {})
        cc = cak.pop("concat_conds")
        kw["cross_attention_kwargs"] = cak or None
        sample = torch.cat([sample, cc.to(sample)], dim=1)
        return orig_forward(sample, timestep, encoder_hidden_states, **kw)

    unet.forward = hooked
    pipe.unet = unet
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _PIPE = pipe
    _VAE = pipe.vae
    return _PIPE


def _encode(img):
    a = np.array(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(a).permute(2, 0, 1)[None].to(device, dtype)
    return _VAE.encode(t).latent_dist.mode() * _VAE.config.scaling_factor


def _fg_on_gray(car_rgba, size=512):
    arr = np.array(car_rgba)
    ys, xs = np.where(arr[:, :, 3] > 10)
    if len(xs):
        car = car_rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    else:
        car = car_rgba
    tw = int(size * 0.82)
    sc = tw / car.width
    if car.height * sc > size * 0.60:
        sc = size * 0.60 / car.height
    nw, nh = max(1, int(car.width * sc)), max(1, int(car.height * sc))
    car = car.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    x = (size - nw) // 2
    y = max(0, int(size * 0.58) - nh)
    canvas.paste(car, (x, y), car)
    return canvas


def _studio_bg(template_key, size=512):
    bg, _, _ = compositing.build_template(template_key)
    return bg.convert("RGB").resize((size, size), Image.LANCZOS)


def relight(car_rgba, template_key):
    pipe = _load()
    fg = _fg_on_gray(car_rgba, 512)
    bg = _studio_bg(template_key, 512)
    fg_latent = _encode(fg)
    bg_latent = _encode(bg)
    cc = torch.cat([fg_latent, bg_latent], dim=1)   # (1, 8, 64, 64)
    cc = torch.cat([cc, cc], dim=0)                  # duplicate for CFG (uncond + cond)

    prompt = PROMPTS.get(template_key, PROMPTS["white_studio"])
    img = pipe(
        prompt=prompt, negative_prompt=NEG,
        width=512, height=512,
        num_inference_steps=STEPS, guidance_scale=GUID,
        cross_attention_kwargs={"concat_conds": cc},
        output_type="pil",
    ).images[0]
    return img.resize((1024, 1024), Image.LANCZOS)
