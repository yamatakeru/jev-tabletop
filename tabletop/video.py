"""Compose the demo video: robot view (left) + what Jev saw/decided (right) + status banner."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
LEFT_W, LEFT_H = 832, 624
PANEL_X = LEFT_W + 12
THUMB = 424
BANNER_Y = LEFT_H


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class VideoComposer:
    def __init__(self, path: Path, fps: int = 25):
        self.fps = fps
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                         macro_block_size=8)
        self.overlay: dict = {}
        self.thumb: Image.Image | None = None
        self.f_title = _font(30)
        self.f_text = _font(19)
        self.f_small = _font(16)

    # ------------------------------------------------------------------ api
    def set_overlay(self, ov: dict) -> None:
        self.overlay = ov
        if ov.get("image_png"):
            img = Image.open(io.BytesIO(base64.b64decode(ov["image_png"]))).convert("RGB")
            self.thumb = img.resize((THUMB, THUMB))

    def frame(self, robot_rgb: np.ndarray, sim_time: float) -> None:
        self.writer.append_data(self.compose(robot_rgb, sim_time))

    def hold(self, robot_rgb: np.ndarray, seconds: float, sim_time: float) -> None:
        f = self.compose(robot_rgb, sim_time)
        for _ in range(int(seconds * self.fps)):
            self.writer.append_data(f)

    def title_card(self, title: str, lines: list[str], seconds: float) -> None:
        img = Image.new("RGB", (W, H), (20, 24, 32))
        d = ImageDraw.Draw(img)
        d.text((80, 220), title, font=_font(56), fill=(255, 255, 255))
        for i, line in enumerate(lines):
            d.text((80, 320 + i * 44), line, font=_font(26), fill=(200, 210, 230))
        arr = np.asarray(img)
        for _ in range(int(seconds * self.fps)):
            self.writer.append_data(arr)

    def close(self) -> None:
        self.writer.close()

    # ------------------------------------------------------------- drawing
    def compose(self, robot_rgb: np.ndarray, sim_time: float) -> np.ndarray:
        canvas = Image.new("RGB", (W, H), (24, 28, 36))
        canvas.paste(Image.fromarray(robot_rgb).resize((LEFT_W, LEFT_H)), (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((12, 8), f"sim t = {sim_time:5.1f} s", font=self.f_small, fill=(30, 30, 30))
        d.text((PANEL_X, 6), self.overlay.get("panel", "What Jev sees (overhead camera)"), font=self.f_small,
               fill=(170, 190, 220))
        if self.thumb is not None:
            canvas.paste(self.thumb, (PANEL_X, 28))
        y = 28 + THUMB + 8
        for line in self.overlay.get("lines", [])[:7]:
            d.text((PANEL_X, y), line, font=self.f_small, fill=(230, 230, 230))
            y += 22
        d.rectangle((0, BANNER_Y, W, H), fill=(14, 16, 22))
        d.text((16, BANNER_Y + 14), self.overlay.get("title", ""), font=self.f_title,
               fill=(255, 220, 120))
        d.text((16, BANNER_Y + 56), self.overlay.get("subtitle", ""), font=self.f_text,
               fill=(200, 210, 230))
        return np.asarray(canvas)
