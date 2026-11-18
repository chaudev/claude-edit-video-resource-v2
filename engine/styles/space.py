"""space: deep-space A roll. Parallax star layers drifting, soft nebula glow, slow planet / warp variants.
Nebula computed at low res and upscaled (cheap); stars drawn at full res so they stay sharp."""
import math, random
import numpy as np
from PIL import Image, ImageDraw
from engine.core import W, H, eo

PAL = {'navy': ((4, 6, 18), (70, 90, 255), (200, 60, 200)),
       'teal': ((2, 12, 18), (0, 180, 200), (60, 90, 255)),
       'red': ((14, 4, 10), (230, 50, 70), (255, 140, 60))}
SW, SH = W // 8, H // 8
_yy, _xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
_r = random.Random(11)
STARS = [(_r.random() * W, _r.random() * H, _r.choice((1, 1, 1, 2, 2, 3)), _r.random() * 6.28) for _ in range(340)]

def _nebula(t, v, seed):
    base, c1, c2 = PAL[v]
    img = np.zeros((SH, SW, 3), np.float32) + np.array(base, np.float32)
    for i, col in enumerate((c1, c2, c1)):
        cx = SW * (.5 + .38 * math.sin(t * (.11 + .03 * i) + i * 2.4 + seed))
        cy = SH * (.5 + .36 * math.cos(t * (.09 + .02 * i) + i * 1.7 + seed))
        r = SW * (.5 + .1 * math.sin(t * .3 + i))
        g = np.exp(-(((_xx - cx) ** 2 + ((_yy - cy) * .6) ** 2) / (r * r)))[..., None]
        img += np.array(col, np.float32) * g * .32
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)

def _stars(img, t, speed=1.0, warp=0.0):
    d = ImageDraw.Draw(img)
    for j, (x, y, s, ph) in enumerate(STARS):
        layer = s  # bigger = nearer = faster
        yy = (y + t * 22 * layer * speed) % H
        xx = (x + t * 6 * layer * speed) % W
        tw = .55 + .45 * math.sin(t * 2.2 + ph)
        c = int(150 + 105 * tw)
        if warp > 0:
            dx, dy = xx - W / 2, yy - H * .45; L = warp * s * 18
            n = math.hypot(dx, dy) + 1
            d.line((xx, yy, xx + dx / n * L, yy + dy / n * L), fill=(c, c, 255), width=s)
        else:
            d.ellipse((xx - s, yy - s, xx + s, yy + s), fill=(c, c, min(255, c + 20)))
    return img

_PL = {}
def _planet_sprite(v, R=250):
    """full shaded sphere (lit from top-left), cached per palette"""
    if v not in _PL:
        _, c1, c2 = PAL[v]
        y, x = np.mgrid[-R:R, -R:R].astype(np.float32) + .5
        d = np.sqrt(x * x + y * y) / R
        nz = np.sqrt(np.clip(1 - d * d, 0, 1))
        lit = np.clip((-x * .55 - y * .55) / R * .9 + nz * .7, 0, 1)[..., None]
        band = (.5 + .5 * np.sin(y / R * 9))[..., None] * .18
        base = np.array(c1, np.float32) * .55 + np.array(c2, np.float32) * band
        rgb = np.clip(base * (.25 + .95 * lit), 0, 255).astype(np.uint8)
        al = (np.clip((1 - d) * R / 2, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(rgb); im.putalpha(Image.fromarray(al)); _PL[v] = im
    return _PL[v]

def _planet(img, t, v):
    _, c1, c2 = PAL[v]; R = 250
    cx, cy = 540 + 40 * math.sin(t * .2), 1380 + 18 * math.cos(t * .17)   # whole planet + ring stay inside the frame
    rx, ry = R * 1.55, 62 + 10 * math.sin(t * .3)
    rc = tuple(int(x * .8) for x in c2) + (255,)
    img = img.convert('RGBA'); d = ImageDraw.Draw(img)
    d.arc((cx - rx, cy - ry, cx + rx, cy + ry), 180, 360, fill=rc, width=9)      # back half of ring
    sp = _planet_sprite(v, R); img.alpha_composite(sp, (int(cx - R), int(cy - R)))
    d = ImageDraw.Draw(img)
    d.arc((cx - rx, cy - ry, cx + rx, cy + ry), 0, 180, fill=rc, width=9)        # front half of ring
    return img.convert('RGB')

def bg(t, variant='stars/navy'):
    kind, v = variant.split('/'); seed = list(PAL).index(v) * 1.9
    img = _nebula(t, v, seed)
    if kind == 'warp': return _stars(img, t, 2.5, warp=1.0 + .4 * math.sin(t * 1.3))
    img = _stars(img, t, 1.0 if kind == 'stars' else .6)
    if kind == 'planet': img = _planet(img, t, v)
    return img

def transition(img, t, dt, variant=None):
    if dt >= .3: return img
    k = 1 - eo(dt / .3); z = 1 + .16 * k; w, h = W / z, H / z
    out = img.resize((W, H), Image.BILINEAR, box=((W - w) / 2, (H - h) / 2, (W + w) / 2, (H + h) / 2))
    if k > .5:
        F = Image.new('RGB', (W, H), (220, 230, 255)); out = Image.blend(out, F, (k - .5) * .5)
    return out

VARIANTS = [f'{k}/{v}' for k in ('stars', 'warp', 'planet') for v in PAL]
