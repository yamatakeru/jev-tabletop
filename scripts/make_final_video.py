"""Assemble the submission video: intro card + demo runs, sped up, with a speed label.

uv run python scripts/make_final_video.py OUT.mp4 --speed 1.5 outputs/demoA/run.mp4 ...
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 25
JP_FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
JP_FONT_LIGHT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def intro(path: Path, seconds: float, speed: float) -> None:
    img = Image.new("RGB", (W, H), (20, 24, 32))
    d = ImageDraw.Draw(img)
    d.text((70, 120), "学習なしで、言葉どおりに片付けるロボット", font=font(JP_FONT, 52),
           fill=(255, 255, 255))
    d.text((70, 200), "Zero-shot tabletop sorting with TypeSafe Jev (System One model)",
           font=font(JP_FONT_LIGHT, 28), fill=(200, 210, 230))
    lines = [
        "・認識と判断はすべて Jev（画像入力）: 次に掴む物 = Choice / 位置 = Box / 成否 = Noul",
        "・動作はスキル（pick / place）+ IK。学習・デモデータは一切なし",
        "・カメラは実機にもある構成のみ: 固定の俯瞰カメラ + ハンドのリストカメラ",
        "・MuJoCo（Franka Panda）+ dora-rs のデータフロー",
        "",
        "Demo 1  明示的な指示     Demo 2  意味的な指示（信号機の色）     Demo 3  外乱からの回復",
    ]
    for i, line in enumerate(lines):
        d.text((70, 290 + i * 48), line, font=font(JP_FONT_LIGHT, 26), fill=(220, 225, 235))
    d.text((70, 640), f"※ 動画は全編 {speed:g} 倍速で再生しています（{speed:g}x speed）",
           font=font(JP_FONT, 24), fill=(255, 220, 120))
    arr = np.asarray(img)
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8,
                            macro_block_size=8) as w:
        for _ in range(int(seconds * FPS * speed)):  # becomes `seconds` after speed-up
            w.append_data(arr)  # pyright: ignore[reportAttributeAccessIssue]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--intro", type=float, default=7.0)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        intro(tdp / "intro.mp4", a.intro, a.speed)
        lst = tdp / "list.txt"
        lst.write_text("".join(f"file '{Path(p).resolve()}'\n"
                               for p in [tdp / "intro.mp4", *a.clips]))
        joined = tdp / "joined.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(lst), "-c", "copy", str(joined)], check=True)
        # Speed label as a PNG overlay (works without ffmpeg's drawtext/freetype).
        label = Image.new("RGBA", (190, 40), (0, 0, 0, 150))
        ImageDraw.Draw(label).text((12, 6), f"{a.speed:g}x speed", font=font(JP_FONT, 22),
                                   fill=(255, 255, 255, 255))
        label.save(tdp / "label.png")
        fc = f"[0:v]setpts=PTS/{a.speed},fps={FPS}[v];[v][1:v]overlay=W-w-14:H-h-12"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-i",
                        str(tdp / "label.png"), "-filter_complex", fc, "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
                       check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
