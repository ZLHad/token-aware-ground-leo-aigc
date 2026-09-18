"""Canonical image resize shared by the 2B scorer (HF) and the 32B answerer (MLX).

Qwen-VL smart_resize: keep aspect ratio, round H/W to multiples of 32 px (patch 16 x merge 2),
and clamp total pixels to [MIN_TOK, MAX_TOK] * 1024 so that n_vis = H*W/1024 in [256, 1024].
Both processors then see an image that needs no further resizing -> identical token grids.
"""
import math
from PIL import Image
FACTOR = 32; MIN_TOK, MAX_TOK = 256, 1024

def _round(x, f=FACTOR): return max(f, int(round(x / f)) * f)
def _floor(x, f=FACTOR): return max(f, int(math.floor(x / f)) * f)
def _ceil(x, f=FACTOR): return int(math.ceil(x / f)) * f

def smart_size(w, h):
    hb, wb = _round(h), _round(w)
    if hb * wb > MAX_TOK * FACTOR * FACTOR:
        beta = math.sqrt((h * w) / (MAX_TOK * FACTOR * FACTOR)); hb, wb = _floor(h / beta), _floor(w / beta)
    elif hb * wb < MIN_TOK * FACTOR * FACTOR:
        beta = math.sqrt((MIN_TOK * FACTOR * FACTOR) / (h * w)); hb, wb = _ceil(h * beta), _ceil(w * beta)
    return wb, hb

def canonical(image: Image.Image) -> Image.Image:
    image = image.convert("RGB"); w, h = smart_size(*image.size)
    return image if image.size == (w, h) else image.resize((w, h), Image.BICUBIC)

def n_tokens(image: Image.Image) -> int:
    w, h = image.size; return (w // FACTOR) * (h // FACTOR)
