"""aurora: soft moving colour blobs (northern-lights feel) over a deep base. Computed at low res, upscaled -> cheap."""
import math
import numpy as np
from PIL import Image
from engine.core import W, H, eo
PAL = {'violet': ((14, 8, 34), [(140, 60, 255), (255, 70, 170), (60, 140, 255)]),
       'ocean': ((4, 16, 30), [(0, 170, 220), (40, 90, 255), (0, 230, 170)]),
       'sunset': ((30, 8, 14), [(255, 90, 60), (255, 170, 40), (200, 40, 120)]),
       'mint': ((6, 24, 22), [(40, 220, 160), (120, 255, 200), (30, 120, 200)])}
VARIANTS = list(PAL)
SW, SH = W // 8, H // 8
_yy, _xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
def bg(t, variant='violet'):
    base, cols = PAL[variant]; seed = VARIANTS.index(variant) * 1.7
    img = np.zeros((SH, SW, 3), np.float32) + np.array(base, np.float32)
    for i in range(5):
        col = np.array(cols[i % 3], np.float32)
        cx = SW * (.5 + .42 * math.sin(t * (.23 + .05 * i) + i * 2.1 + seed))
        cy = SH * (.5 + .40 * math.cos(t * (.17 + .04 * i) + i * 1.3 + seed))
        r = SW * (.55 + .15 * math.sin(t * .4 + i))
        g = np.exp(-(((_xx - cx) ** 2 + ((_yy - cy) * .7) ** 2) / (r * r)))[..., None]
        img += col * g * .55
    # slow diagonal ribbon
    rib = np.exp(-((_yy - SH * .5 - (_xx - SW / 2) * .6 - 40 * math.sin(t * .5 + seed)) ** 2) / 300)[..., None]
    img += np.array(cols[0], np.float32) * rib * .35
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
def transition(img, t, dt, variant=None):
    if dt >= .28: return img
    k = 1 - eo(dt / .28); z = 1 + .12 * k; w, h = W / z, H / z
    return img.resize((W, H), Image.BILINEAR, box=((W - w) / 2, (H - h) / 2, (W + w) / 2, (H + h) / 2))
