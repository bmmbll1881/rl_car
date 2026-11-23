"""Training / evaluation visuals drawn with Pillow (no matplotlib required)."""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = {
    "dqn": (214, 69, 65),
    "ppo": (44, 123, 182),
    "expert": (46, 160, 67),
    "random": (128, 128, 128),
}
GRID = (226, 229, 233)
AXIS = (60, 65, 70)
TEXT = (35, 40, 45)


def _font(size: int = 14, bold: bool = False):
    names = ["segoeuib.ttf", "segoeui.ttf", "arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]
    if not bold:
        names = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ------------------------------------------------------------------ frame render
def render_frame(env, scale: float = 4.5, show_lidar: bool = True, hud: str | None = None,
                 trail: list | None = None) -> Image.Image:
    """Top-down picture of the current environment state."""
    tr = env.track
    x0, y0 = tr.bbox[0]
    x1, y1 = tr.bbox[1]
    pad = 6
    W = int((x1 - x0 + 2 * pad) * scale)
    H = int((y1 - y0 + 2 * pad) * scale)
    img = Image.new("RGB", (W, H), (247, 248, 250))
    dr = ImageDraw.Draw(img)

    def to_px(p):
        return ((p[0] - x0 + pad) * scale, H - (p[1] - y0 + pad) * scale)

    # road surface
    left = [to_px(p) for p in tr.left]
    right = [to_px(p) for p in tr.right][::-1]
    dr.polygon(left + right, fill=(189, 195, 199), outline=(150, 156, 161))

    # lane divider markings (dashed), computed from the lane layout
    centers = getattr(env, "lane_centers", None)
    if centers is not None and len(centers) > 1:
        bounds = [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]
        for b in bounds:
            line = tr.points + tr.normals * b
            for i in range(0, tr.n_points, 6):
                a = to_px(line[i])
                c = to_px(line[(i + 3) % tr.n_points])
                dr.line([a, c], fill=(238, 238, 226), width=1)
    else:
        for i in range(0, tr.n_points, 6):
            a = to_px(tr.points[i])
            b2 = to_px(tr.points[(i + 3) % tr.n_points])
            dr.line([a, b2], fill=(235, 235, 225), width=1)

    # start / finish line
    a = to_px(tr.left[0])
    b = to_px(tr.right[0])
    dr.line([a, b], fill=(240, 240, 240), width=3)

    if trail:
        pts = [to_px(p) for p in trail[::2]]
        if len(pts) > 1:
            dr.line(pts, fill=(52, 101, 164), width=2)

    # lidar rays
    if show_lidar:
        ranges = tr.raycast((env.x, env.y), env.theta + env.ray_angles, max_range=env.lidar_range, hint=env.idx)
        for ang, rng in zip(env.ray_angles, ranges):
            p = to_px((env.x + rng * math.cos(env.theta + ang), env.y + rng * math.sin(env.theta + ang)))
            dr.line([to_px((env.x, env.y)), p], fill=(255, 178, 76), width=1)

    # other vehicles (the RARL adversary, i.e. vehicle 0, is drawn in red)
    ob = getattr(env, "obstacles", None)
    if ob is not None and getattr(ob, "n", 0):
        adversary_active = getattr(env, "_adversary_hook", None) is not None
        for i, (pos, hd) in enumerate(zip(ob.pos, ob.heading)):
            corners_local = np.array([[-ob.length / 2, -ob.width / 2], [ob.length / 2, -ob.width / 2],
                                      [ob.length / 2, ob.width / 2], [-ob.length / 2, ob.width / 2]])
            c2, s2 = math.cos(hd), math.sin(hd)
            rot2 = np.array([[c2, -s2], [s2, c2]])
            corners = [to_px(pos + rot2 @ p) for p in corners_local]
            if adversary_active and i == 0:
                dr.polygon(corners, fill=(198, 60, 60), outline=(120, 30, 30))
            else:
                dr.polygon(corners, fill=(196, 122, 44), outline=(120, 70, 20))
            nose = to_px((pos[0] + 2.0 * c2, pos[1] + 2.0 * s2))
            dr.line([to_px(pos), nose], fill=(255, 240, 220), width=1)

    # car body
    L, Wd = 4.2, 2.0
    corners_local = np.array([[-L / 2, -Wd / 2], [L / 2, -Wd / 2], [L / 2, Wd / 2], [-L / 2, Wd / 2]])
    c, s = math.cos(env.theta), math.sin(env.theta)
    rot = np.array([[c, -s], [s, c]])
    corners = [to_px((env.x, env.y) + rot @ p) for p in corners_local]
    dr.polygon(corners, fill=(41, 128, 185), outline=(21, 67, 96))
    nose = to_px((env.x + 2.6 * c, env.y + 2.6 * s))
    dr.line([to_px((env.x, env.y)), nose], fill=(250, 250, 250), width=2)

    if hud:
        dr.rectangle([4, 4, 4 + 8 * len(hud) + 8, 26], fill=(255, 255, 255))
        dr.text((8, 7), hud, fill=TEXT, font=_font(13))
    return img


def make_gif(frames, path: str, duration: int = 70, resize_to: int = 900):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for fr in frames:
        im = fr.convert("P", palette=Image.ADAPTIVE, colors=128)
        if im.width > resize_to:
            im = im.resize((resize_to, int(im.height * resize_to / im.width)), Image.LANCZOS)
        out.append(im)
    out[0].save(path, save_all=True, append_images=out[1:], duration=duration, loop=0, optimize=True)
    return path


# ------------------------------------------------------------------- chart helpers
def _nice_ticks(lo, hi, n=5):
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-12:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def line_chart(path: str, series, title: str = "", xlabel: str = "", ylabel: str = "",
               width: int = 860, height: int = 460, ylim=None, xlim=None, hlines=(), legend=True,
               pad_frac: float = 0.06):
    """``series``: list of dict(x=array, y=array, label=str, color=(r,g,b), band=(low, high)).

    ``pad_frac`` widens the y range on both sides; set it to 0 when the caller
    passes an exact ``ylim`` (e.g. a rate that must stay inside [0, 1]).
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f_t, f_l = _font(17, True), _font(13)
    left, right, top, bot = 78, 24, 56, 62

    xs = np.asarray(np.concatenate([np.asarray(s["x"], float)
                                    for s in series] + ([np.asarray(xlim, float)] if xlim else [])), float)
    ys = np.asarray(np.concatenate([np.asarray(s["y"], float)
                                    for s in series] + ([np.asarray(ylim, float)] if ylim else [])), float)
    for s in series:
        if s.get("band") is not None:
            ys = np.concatenate([ys, np.asarray(s["band"][0], float), np.asarray(s["band"][1], float)])
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if xs.size == 0:
        xs = np.array([0.0, 1.0])
    if ys.size == 0:
        ys = np.array([0.0, 1.0])
    x_lo, x_hi = (xlim if xlim else (float(np.min(xs)), float(np.max(xs))))
    y_lo, y_hi = (ylim if ylim else (float(np.min(ys)), float(np.max(ys))))
    if y_lo > 0:
        y_lo = 0.0
    pad = float(pad_frac) * (y_hi - y_lo + 1e-9)
    if float(np.min(ys)) >= 0.0 or (ylim and ylim[0] is not None and float(ylim[0]) >= 0.0):
        # never draw a negative axis for non-negative data (rates, speeds, ...)
        y_lo = max(0.0, y_lo - pad)
    else:
        y_lo -= pad
    y_hi += pad

    def px(x):
        return left + (x - x_lo) / (x_hi - x_lo + 1e-12) * (width - left - right)

    def py(y):
        return height - bot - (y - y_lo) / (y_hi - y_lo + 1e-12) * (height - top - bot)

    for t in _nice_ticks(y_lo, y_hi, 6):
        y = py(t)
        dr.line([(left, y), (width - right, y)], fill=GRID)
        dr.text((left - 8, y - 7), f"{t:g}", fill=TEXT, font=f_l, anchor="ra")
    for t in _nice_ticks(x_lo, x_hi, 6):
        x = px(t)
        dr.line([(x, top), (x, height - bot)], fill=GRID)
        dr.text((x, height - bot + 8), f"{t:g}", fill=TEXT, font=f_l, anchor="ma")
    dr.line([(left, top), (left, height - bot), (width - right, height - bot)], fill=AXIS, width=2)

    for h in hlines:
        y = py(h["y"])
        dr.line([(left, y), (width - right, y)], fill=h.get("color", (150, 150, 150)), width=2)

    for s in series:
        x = np.asarray(s["x"], float)
        y = np.asarray(s["y"], float)
        color = s.get("color", (44, 123, 182))
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if s.get("band") is not None and len(x):
            lo, hi = np.asarray(s["band"][0], float)[ok], np.asarray(s["band"][1], float)[ok]
            good = np.isfinite(lo) & np.isfinite(hi)
            poly = ([(px(a), py(b)) for a, b in zip(x[good], hi[good])] +
                    [(px(a), py(b)) for a, b in zip(x[good][::-1], lo[good][::-1])])
            if len(poly) > 2:
                dr.polygon(poly, fill=tuple(int(c + (255 - c) * 0.78) for c in color))
        pts = [(px(a), py(b)) for a, b in zip(x, y)]
        if len(pts) > 1:
            dr.line(pts, fill=color, width=3 if s.get("bold", True) else 2)

    # reference labels go on top of the data, otherwise a band hides them
    for h in hlines:
        if h.get("label"):
            y = py(h["y"])
            dr.text((width - right - 4, y - 16), h["label"], fill=h.get("color", (120, 120, 120)), font=f_l, anchor="ra")

    if title:
        dr.text((left, 18), title, fill=TEXT, font=f_t)
    if xlabel:
        dr.text(((left + width - right) / 2, height - 26), xlabel, fill=TEXT, font=f_l, anchor="ma")
    if ylabel:
        tmp = Image.new("RGBA", (len(ylabel) * 9 + 8, 22), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((2, 2), ylabel, fill=TEXT, font=f_l)
        tmp = tmp.rotate(90, expand=True)
        img.paste(tmp, (8, int((top + height - bot) / 2 - tmp.height / 2)), tmp)

    if legend:
        entries = [s for s in series if s.get("label")]
        if entries:
            bw = 34 + sum(int(7.2 * len(s["label"])) + 34 for s in entries) + 10
            bh = 22
            dr.rectangle([left + 6, top + 4, left + 6 + bw, top + 4 + bh], fill=(255, 255, 255),
                         outline=(225, 228, 232))
            x = left + 14
            y = top + 9
            for s in entries:
                dr.line([(x, y + 7), (x + 22, y + 7)], fill=s.get("color", (44, 123, 182)), width=3)
                dr.text((x + 28, y), s["label"], fill=TEXT, font=f_l)
                x += 34 + int(7.2 * len(s["label"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


def bar_chart(path: str, groups, title: str = "", ylabel: str = "", width: int = 760, height: int = 400):
    """``groups``: list of dict(label=str, values=[(name, value, color)], ymax=float)."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f_t, f_l = _font(16, True), _font(13)
    left, right, top, bot = 74, 24, 84, 62

    n_series = len(groups[0]["values"])
    ymax = max(g["ymax"] for g in groups)
    ymax = ymax * 1.08 + 1e-9

    def px(i):
        slot = (width - left - right) / len(groups)
        return left + slot * (i + 0.5)

    def py(v):
        return height - bot - v / ymax * (height - top - bot)

    for t in _nice_ticks(0, ymax, 5):
        y = py(t)
        dr.line([(left, y), (width - right, y)], fill=GRID)
        dr.text((left - 8, y - 7), f"{t:g}", fill=TEXT, font=f_l, anchor="ra")
    dr.line([(left, top), (left, height - bot), (width - right, height - bot)], fill=AXIS, width=2)

    slot = (width - left - right) / len(groups)
    bw = slot * 0.62 / n_series
    for gi, g in enumerate(groups):
        for si, (name, val, color) in enumerate(g["values"]):
            if val is None or not math.isfinite(float(val)):
                continue
            x0 = px(gi) - slot * 0.31 + si * bw
            dr.rectangle([x0, py(val), x0 + bw * 0.92, height - bot], fill=color)
            dr.text((x0 + bw * 0.46, py(val) - 16), f"{val:.3g}", fill=TEXT, font=f_l, anchor="ma")
        dr.text((px(gi), height - bot + 10), g["label"], fill=TEXT, font=f_l, anchor="ma")

    x = left + 12
    for name, _, color in groups[0]["values"]:
        dr.rectangle([x, 44, x + 14, 58], fill=color)
        dr.text((x + 20, 42), name, fill=TEXT, font=f_l)
        x += 34 + int(7.2 * len(name))
    if title:
        dr.text((left, 14), title, fill=TEXT, font=f_t)
    if ylabel:
        tmp = Image.new("RGBA", (len(ylabel) * 9 + 8, 22), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((2, 2), ylabel, fill=TEXT, font=f_l)
        tmp = tmp.rotate(90, expand=True)
        img.paste(tmp, (8, int((top + height - bot) / 2 - tmp.height / 2)), tmp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path
