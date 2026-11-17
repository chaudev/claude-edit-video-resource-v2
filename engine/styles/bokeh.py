"""bokeh: dark gradient with soft out-of-focus light orbs drifting upward."""
import math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from engine.core import W, H, eo
PAL = {'gold': ((18, 12, 8), (40, 26, 14), [(255, 190, 90), (255, 140, 60), (255, 230, 170)]),
       'neon': ((8, 8, 24), (26, 14, 48), [(255, 70, 160), (80, 160, 255), (170, 90, 255)]),
       'teal': ((4, 18, 22), (10, 40, 46), [(60, 220, 210), (120, 200, 255), (200, 255, 230)]),
       'ruby': ((22, 4, 10), (50, 10, 22), [(255, 70, 90), (255, 140, 120), (255, 200, 160)])}
VARIANTS = list(PAL)
_BASE = {}; _SPR = {}
def _base(v):
    if v not in _BASE:
        a, b, _ = PAL[v]; k = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        _BASE[v] = Image.fromarray((np.array(a) * (1 - k) + np.array(b) * k).repeat(W, 1).astype(np.uint8))
    return _BASE[v]
def _sprite(col, r):
    key = (col, r)
    if key not in _SPR:
        L = Image.new('RGBA', (r * 4, r * 4), col + (0,)); d = ImageDraw.Draw(L)
        d.ellipse((r, r, 3 * r, 3 * r), fill=col + (120,)); _SPR[key] = L.filter(ImageFilter.GaussianBlur(r * .35))
    return _SPR[key]
_rng = random.Random(11)
ORBS = [(_rng.uniform(0, W), _rng.uniform(0, H), _rng.choice([30, 45, 60, 90, 120]), _rng.randrange(3), _rng.uniform(20, 70), _rng.uniform(0, 6.3)) for _ in range(34)]
def bg(t, variant='gold'):
    img = _base(variant).convert('RGBA'); cols = PAL[variant][2]
    for x, y, r, ci, sp, ph in ORBS:
        yy = (y - t * sp) % (H + 4 * r) - 2 * r; xx = x + 40 * math.sin(t * .5 + ph)
        s = _sprite(cols[ci], r); a = .55 + .45 * math.sin(t * 1.3 + ph)
        if a < 1:
            s = s.copy(); s.putalpha(s.getchannel('A').point(lambda v: int(v * a)))
        img.alpha_composite(s, (int(xx - 2 * r), int(yy - 2 * r))) if 0 <= xx - 2 * r and xx + 2 * r < W and 0 <= yy - 2 * r and yy + 2 * r < H else img.paste(s, (int(xx - 2 * r), int(yy - 2 * r)), s)
    return img.convert('RGB')
def transition(img, t, dt, variant=None):
    if dt >= .28: return img
    k = 1 - eo(dt / .28); z = 1 + .12 * k; w, h = W / z, H / z
    return img.resize((W, H), Image.BILINEAR, box=((W - w) / 2, (H - h) / 2, (W + w) / 2, (H + h) / 2))
