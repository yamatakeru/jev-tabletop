"""Coarse-to-fine localisation with Jev Box queries on zoomed crops."""

from __future__ import annotations

import io
from collections.abc import Callable

from PIL import Image

Box = tuple[float, float, float, float]


def crop_window(box: Box, size: float) -> tuple[float, float]:
    cu, cv = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    l0 = min(max(cu - size / 2, 0.0), 1.0 - size)
    t0 = min(max(cv - size / 2, 0.0), 1.0 - size)
    return l0, t0


def crop_png(img: Image.Image, l0: float, t0: float, size: float, out: int = 512) -> tuple[bytes, Image.Image]:
    S = img.size[0]
    c = img.crop((int(l0 * S), int(t0 * S), int((l0 + size) * S), int((t0 + size) * S)))
    c = c.resize((out, out))
    buf = io.BytesIO()
    c.save(buf, format="PNG")
    return buf.getvalue(), c


def to_full(b: Box, l0: float, t0: float, size: float) -> Box:
    return (l0 + b[0] * size, t0 + b[1] * size, l0 + b[2] * size, t0 + b[3] * size)


def to_crop(b: Box, l0: float, t0: float, size: float) -> Box:
    return ((b[0] - l0) / size, (b[1] - t0) / size, (b[2] - l0) / size, (b[3] - t0) / size)


def coarse_to_fine(
    img: Image.Image,
    coarse: Box,
    ask: Callable[[bytes, Box | None], dict],
    gt: Box | None = None,
    rounds: int = 2,
) -> tuple[Box, list[dict]]:
    """Zoom in around the current estimate and re-ask Box. The first window adapts to the
    coarse box size (a wide/uncertain box gets a wider crop); later windows shrink."""
    box = coarse
    trace = []
    for r in range(rounds):
        w = max(box[2] - box[0], box[3] - box[1])
        size = min(max(w + 0.14, 0.2), 0.5) if r == 0 else 0.14
        l0, t0 = crop_window(box, size)
        png, cimg = crop_png(img, l0, t0, size)
        ans = ask(png, to_crop(gt, l0, t0, size) if gt else None)
        box = to_full(ans["box"], l0, t0, size)
        trace.append({"round": r, "window": [l0, t0, size], "crop_box": ans["box"],
                      "conf": ans.get("conf"), "visible": ans.get("visible"),
                      "latency_s": ans.get("latency_s"), "crop_img": cimg})
    return box, trace
