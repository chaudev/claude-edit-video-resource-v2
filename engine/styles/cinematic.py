"""cinematic: footage-driven style. Each scene plays its own real video clip (16:9 source CROPPED to 9:16, never squeezed),
teal/orange grade, slow push-in, film grain, vignette, warm light-leak transitions. Serif titles (Lora) + tracked gold kickers.
Usage: from engine.styles import cinematic as S ; video, SFX = S.make(VOICE, 'words.json', SPECS, footage_dir)
SPEC = dict(t=start_seconds, clip=('f13.mp4', src_start, xfrac), type='T'|'N'|'V'|'P'|'Q'|'CTA', ...)"""
import math, subprocess, random, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from engine.core import W, H, FPS, SAFE, font, clamp, prog, eo, eio, back, lerp, paste_alpha, tx, rect, chip, reg, ap, pop, punch, counter, logo_cta, Video

CREAM = (246, 238, 224); GOLD = (236, 192, 112); TEAL = (124, 206, 214); RED = (232, 82, 64); INK = (10, 10, 14)
def serif(s): return font('lora', s, 'Bold')
def serifi(s): return font('lorai', s, 'SemiBold')
def sans(s, w='SemiBold'): return font('bv', s, w)
def an(s): return font('anton', s)
VARIANTS = ['footage']
BW, BH = 1188, 2112                       # footage decoded 10% larger -> room for push-in / drift
GRADE = ("eq=contrast=1.12:saturation=0.80:brightness=-0.06:gamma=0.94,"
         "colorbalance=rs=-0.06:gs=-0.02:bs=0.08:rh=0.08:gh=0.02:bh=-0.07,unsharp=5:5:0.5")

class Clip:
    """sequential raw-frame reader of a cropped+graded footage clip"""
    def __init__(self, path, start, xfrac=.5, dur=6, zm=1.0, yf=.5):
        self.path, self.start, self.x, self.dur, self.zm, self.yf = path, start, xfrac, dur, zm, yf
        self.p = None; self.i = -1; self.last = None
    def _open(self, i):
        if self.p: self.p.kill()
        z = self.zm; vf = f"crop=ih*9/16/{z}:ih/{z}:(iw-ih*9/16/{z})*{self.x}:(ih-ih/{z})*{self.yf},scale={BW}:{BH}:flags=lanczos,fps={FPS},{GRADE}"
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{self.start + i / FPS:.3f}', '-i', self.path, '-t', f'{self.dur + 1:.2f}',
                                   '-vf', vf, '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.i = i - 1
    def frame(self, i):
        if os.environ.get('NOFOOT'): return Image.new('RGB', (BW, BH), (40, 60, 70))
        if self.p is None or i <= self.i - 1 or i > self.i + 45: self._open(i)
        if i == self.i and self.last is not None: return Image.frombuffer('RGB', (BW, BH), self.last)
        while self.i < i:
            b = self.p.stdout.read(BW * BH * 3)
            if len(b) < BW * BH * 3: break
            self.last = b; self.i += 1
        if self.last is None: return Image.new('RGB', (BW, BH), INK)
        return Image.frombuffer('RGB', (BW, BH), self.last)

# ---------- overlays ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_v = ((xx - W / 2) / (W * .62)) ** 2 + ((yy - H * .46) / (H * .58)) ** 2
VIG = Image.new('RGBA', (W, H), (4, 6, 10, 0)); VIG.putalpha(Image.fromarray((np.clip(_v - .35, 0, 1) ** 1.3 * 215).astype(np.uint8)))
_k = np.clip((yy[:, :1] - 1280) / (H - 1280), 0, 1) ** 1.6
SHADE = Image.new('RGBA', (W, H), (0, 4, 12, 0)); SHADE.putalpha(Image.fromarray(np.repeat((_k * 170).astype(np.uint8), W, 1)))
_r = random.Random(3); GRAIN = []
for s in range(6):
    n = np.random.RandomState(s).normal(0, 1, (H // 3, W // 3))
    a = np.clip(np.abs(n) * 26, 0, 60).astype(np.uint8); g = Image.new('RGBA', (W // 3, H // 3), (255, 255, 255, 0))
    g.putalpha(Image.fromarray(a)); GRAIN.append(g.resize((W, H), Image.NEAREST))
def leak(col, cx, cy, r):
    L = Image.new('RGBA', (W, H), col + (0,))
    d = np.sqrt(((xx - cx) / r) ** 2 + ((yy - cy) / r) ** 2)
    L.putalpha(Image.fromarray((np.clip(1 - d, 0, 1) ** 2 * 255).astype(np.uint8))); return L
LEAKS = [leak((255, 150, 60), 1000, 300, 900), leak((255, 90, 50), 60, 1500, 1000), leak((255, 210, 140), 540, 900, 1200)]
_ws = {}
def scrim(y0, y1, a=150):
    """soft dark haze behind a text block (feathered, never a hard band)"""
    k = (y0, y1, a)
    if k not in _ws:
        c = (y0 + y1) / 2; h = (y1 - y0) / 2 + 260
        v = np.clip(1 - np.abs(yy[:, :1] - c) / h, 0, 1) ** 1.4 * a
        L = Image.new('RGBA', (W, H), (4, 6, 10, 0)); L.putalpha(Image.fromarray(np.repeat(v.astype(np.uint8), W, 1))); _ws[k] = L
    return _ws[k]

def footage_bg(clip, t, t0, d, seed):
    i = int(round((t - t0) * FPS)); im = clip.frame(max(0, i))
    k = clamp((t - t0) / max(d, 1)); z = 1.0 + .09 * eio(k)          # slow push-in
    w, h = BW / z * (1080 / 1080), BH / z
    w = max(W, BW / z); h = w * H / W
    dx = (BW - w) * (.5 + .35 * math.sin(seed * 2.1)); dy = (BH - h) * (.5 + .3 * math.cos(seed * 1.3))
    im = im.resize((W, H), Image.BILINEAR, box=(dx, dy, dx + w, dy + h))
    im = im.convert('RGBA')
    g = GRAIN[int(t * 24) % 6]; im.alpha_composite(g)
    lk = LEAKS[seed % 3].copy(); fl = .10 + .06 * math.sin(t * 1.7 + seed)
    lk.putalpha(lk.getchannel('A').point(lambda v: int(v * fl))); im.alpha_composite(lk)
    im.alpha_composite(VIG); im.alpha_composite(SHADE)
    return im.convert('RGB')

def transition(img, t, dt, si):
    """warm light-leak flash burning into the new shot"""
    if dt >= .35: return img
    k = 1 - eo(dt / .35)
    lk = LEAKS[(si + 2) % 3].copy(); lk.putalpha(lk.getchannel('A').point(lambda v: int(v * k * .95)))
    im = img.convert('RGBA'); im.alpha_composite(lk)
    if k > .55:
        wf = Image.new('RGBA', (W, H), (255, 236, 210, int((k - .55) / .45 * 170))); im.alpha_composite(wf)
    return im.convert('RGB')

# ---------- text building blocks ----------
def fit(fn, text, size, maxw=SAFE[2] - SAFE[0] - 30):
    f = fn(size)
    while f.getlength(text) > maxw and size > 20: size -= 4; f = fn(size)
    return f
def track(s): return '   '.join(' '.join(w) for w in s.split(' '))
def kicker(img, text, y, t, at, col=GOLD):
    k, dy = ap(t, at, .45)
    tx(img, track(text.upper()), fit(lambda s: sans(s), track(text.upper()), 36), 540, y, col, k, dy=dy * .5, shadow=INK)
def rule(img, y, t, at, w=220, col=GOLD):
    k = eio(prog(t, at, .6))
    if k > 0: rect(img, (540 - w * k / 2, y - 2, 540 + w * k / 2 + 1, y + 2), col, alpha=k, group=False)
def line(img, text, y, t, at, size=96, col=CREAM, fn=serif):
    p = prog(t, at, .55)
    if p <= 0: return
    f = fit(fn, text, size); s = 1.06 - .06 * eo(p)
    tx(img, text, f, 540, y, col, eo(p), scale=s, dy=(1 - eo(p)) * 26, shadow=INK)

# ---------- scene renderers ----------
def scene_fn(sp, nxt):
    ty = sp.get('type', 'T'); t0 = sp['t']; Y = sp.get('y', 820)
    def T(img, t, _):
        lines = sp.get('l', []); n = len(lines); size = sp.get('size', 118 if n < 3 else 100); lh = size * 1.2
        y0 = Y - (n - 1) * lh / 2
        if sp.get('k'): kicker(img, sp['k'], y0 - size * .5 - 70, t, t0 + .08)
        for j, s in enumerate(lines):
            col = GOLD if j in sp.get('hi', ()) else CREAM
            line(img, s, y0 + j * lh, t, sp.get('at', [t0 + .2 + .22 * j] * n)[j] if 'at' in sp else t0 + .2 + .22 * j, size, col)
        rule(img, y0 + (n - 1) * lh + size * .5 + 56, t, t0 + .35)
        if sp.get('sub'):
            k, dy = ap(t, sp.get('sub_at', t0 + .7), .5)
            tx(img, sp['sub'], fit(serifi, sp['sub'], 62), 540, y0 + (n - 1) * lh + size * .5 + 130, TEAL, k, dy=dy, shadow=INK)
    def N(img, t, _):
        v0, v1, fmt = sp['n']; a = sp.get('n_at', t0 + .2)
        if sp.get('k'): kicker(img, sp['k'], Y - 190, t, t0 + .05)
        p = prog(t, a, .3)
        if p > 0:
            s = counter(t, a, sp.get('n_d', 1.2), v0, v1, '{:,}')
            txt = fmt.format(s); f = fit(an, fmt.format(counter(1, 0, 1, v0, v1, '{:,}')), sp.get('size', 230))
            tx(img, txt, f, 540, Y, GOLD, eo(p), scale=1.08 - .08 * eo(p), shadow=INK)
        rule(img, Y + 150, t, a + .2)
        for j, s in enumerate(sp.get('l', [])):
            line(img, s, Y + 240 + j * 92, t, a + .35 + .2 * j, 78, CREAM, serif)
    def V(img, t, _):
        (a1, u1, b1), (a2, u2, b2) = sp['v']; at = sp.get('v_at', (t0 + .2, t0 + 1.0))
        if sp.get('k'): kicker(img, sp['k'], Y - 250, t, t0 + .05)
        f = fit(an, max(a1, a2, key=len), sp.get('size', 230), 400)
        for j, (val, unit, lab, x, col) in enumerate([(a1, u1, b1, 290, CREAM), (a2, u2, b2, 790, GOLD)]):
            p = prog(t, at[j], .4)
            if p <= 0: continue
            tx(img, val, f, x, Y, col, eo(p), scale=1.1 - .1 * eo(p), shadow=INK, group=('v', j))
            k, dy = ap(t, at[j] + .15, .4)
            tx(img, unit.upper(), sans(46, 'ExtraBold'), x, Y + 150, col, k, dy=dy, shadow=INK, group=('vu', j))
            tx(img, lab, fit(serif, lab, 56, 420), x, Y + 225, CREAM, k, dy=dy, shadow=INK, group=('vl', j))
        k = eio(prog(t, at[0] + .1, .5))
        if k > 0: rect(img, (538, Y - 130 * k, 542, Y + 250 * k), GOLD, alpha=k, group=False)
        if sp.get('l'):
            for j, s in enumerate(sp['l']): line(img, s, Y + 360 + j * 96, t, at[1] + .5 + .2 * j, 76, TEAL if j == 0 else CREAM, serifi)
    def P(img, t, _):
        if sp.get('k'): kicker(img, sp['k'], Y - 200, t, t0 + .05)
        for j, (s, at) in enumerate(sp['p']):
            col = GOLD if j in sp.get('hi', ()) else CREAM
            f = fit(an if sp.get('font') != 'serif' else serif, s, sp.get('size', 160))
            punch(img, s, f, 540, Y + j * sp.get('lh', 240), col, t, at, glow=(255, 160, 80))
        if sp.get('sub'):
            k, dy = ap(t, sp.get('sub_at', t0 + 1), .5)
            yb = Y + (len(sp['p']) - 1) * sp.get('lh', 240) + 150
            tx(img, sp['sub'], fit(serifi, sp['sub'], 58), 540, yb, TEAL, k, dy=dy, shadow=INK)
    def Q(img, t, _):
        lines = sp['q']; n = len(lines); lh = 118; y0 = Y - (n - 1) * lh / 2
        k, _ = ap(t, t0 + .05, .5); tx(img, '“', serif(220), 540, y0 - 150, GOLD, k * .9, group=False)
        for j, s in enumerate(lines):
            line(img, s, y0 + j * lh, t, t0 + .25 + .3 * j, 92, GOLD if j in sp.get('hi', ()) else CREAM, serifi)
        rule(img, y0 + (n - 1) * lh + 110, t, t0 + .6)
    def CTA(img, t, _):
        if sp.get('q'):
            k, dy = ap(t, t0 + .05, .5)
            tx(img, sp['q'], fit(serifi, sp['q'], 70), 540, 300, GOLD, k, dy=dy, shadow=INK)
        logo_cta(img, t, sp['logo_at'], serif(76), CREAM, GOLD, sans(46, 'ExtraBold'), INK, GOLD, sp['follow_at'], y_logo=sp.get('y_logo', 720))
    fn = dict(T=T, N=N, V=V, P=P, Q=Q, CTA=CTA)[ty]
    if sp.get('scrim', False):
        rng = {'T': (Y - 260, Y + 200), 'N': (Y - 230, Y + 330), 'V': (Y - 280, Y + 400), 'P': (Y - 220, Y + 240),
               'Q': (Y - 260, Y + 200), 'CTA': (250, 1250)}[ty]
        def wrapped(img, t, t0_, _f=fn, _r=rng):
            a = eo(prog(t, sp['t'], .4)); L = scrim(*_r, 195)
            if a < 1: L = L.copy(); L.putalpha(L.getchannel('A').point(lambda v: int(v * a)))
            img.paste(L, (0, 0), L); _f(img, t, t0_)
        return wrapped
    return fn

def make(voice, specs, dur, ftdir, shakes=(), sfx_extra=()):
    specs = sorted(specs, key=lambda s: s['t'])
    clips = []
    for j, sp in enumerate(specs):
        d = (specs[j + 1]['t'] if j + 1 < len(specs) else dur) - sp['t']
        f, st, x = sp['clip'][:3]; zm, yf = (sp['clip'][3:] + (1.0, .5))[:2] if len(sp['clip']) > 3 else (1.0, .5)
        clips.append((Clip(ftdir + f, st, x, d + .5, zm, yf), d))
    scenes = [(sp['t'], scene_fn(sp, None)) for sp in specs]
    starts = [sp['t'] for sp in specs]
    bgfn = lambda t, si: footage_bg(clips[si][0], t, starts[si], clips[si][1], si * 7 + 3)
    video = Video(scenes, bgfn, dur, voice, transition=lambda img, t, dt, si: transition(img, t, dt, si), shakes=list(shakes))
    sfx = [(0, 'riser', .35)] + [(a - .08, 'whoosh_s', .32) for a in starts[1:]]
    for sp in specs:
        if sp.get('type') == 'P':
            for s, at in sp['p']: sfx.append((at - .03, 'impact', .7))
        if sp.get('type') == 'N': sfx.append((sp.get('n_at', sp['t'] + .2) + sp.get('n_d', 1.2) - .1, 'ding_lo', .45))
        if sp.get('type') == 'V':
            for a in sp.get('v_at', (sp['t'] + .2, sp['t'] + 1.0)): sfx.append((a, 'thud', .55))
        if sp.get('type') == 'CTA': sfx.append((sp['follow_at'], 'pop_hi', .5))
    return video, sorted(sfx + list(sfx_extra))
