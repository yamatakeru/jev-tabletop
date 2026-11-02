"""Assemble the submission video: intro card + demo runs, sped up, with a speed label.

uv run python scripts/make_final_video.py OUT.mp4 --speed 1.5 [--cards cards.json] \
    [--op OP.mp4 --ed ED.mp4] outputs/demo_ov/demo1/run.mp4 ...

--cards: JSON list (one entry per clip) of {"title": str, "lines": [str, ...]}. The title card
recorded at the start of each run.mp4 (4 s, Latin-only font) is cut and replaced by this card.
"""

from __future__ import annotations

import argparse
import json
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


RUN_CARD_S = 4.0  # length of the title card that nodes/sim.py records at the start of run.mp4


def fit(d: ImageDraw.ImageDraw, text: str, path: str, size: int, width: int):
    f = font(path, size)
    while d.textlength(text, font=f) > width and size > 12:
        size -= 2
        f = font(path, size)
    return f


def still(path: Path, img: Image.Image, seconds: float, speed: float) -> None:
    arr = np.asarray(img)
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8,
                            macro_block_size=8) as w:
        for _ in range(int(seconds * FPS * speed)):  # becomes `seconds` after speed-up
            w.append_data(arr)  # pyright: ignore[reportAttributeAccessIssue]


def card(path: Path, title: str, lines: list[str], seconds: float, speed: float) -> None:
    img = Image.new("RGB", (W, H), (20, 24, 32))
    d = ImageDraw.Draw(img)
    d.text((80, 220), title, font=fit(d, title, JP_FONT, 48, W - 160), fill=(255, 255, 255))
    for i, line in enumerate(lines):
        d.text((80, 315 + i * 46), line, font=fit(d, line, JP_FONT_LIGHT, 26, W - 160),
               fill=(205, 212, 228))
    still(path, img, seconds, speed)


def trim(src: str, dst: Path, start: float) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-i", src, "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS),
                    str(dst)], check=True)


def intro(path: Path, seconds: float, speed: float) -> None:
    img = Image.new("RGB", (W, H), (20, 24, 32))
    d = ImageDraw.Draw(img)
    title = "System One モデル Jev による学習なしの物体片付けロボット"
    d.text((70, 110), title, font=fit(d, title, JP_FONT, 44, W - 140), fill=(255, 255, 255))
    d.text((70, 180), "Training-free tabletop tidying with TypeSafe Jev (System One model)",
           font=font(JP_FONT_LIGHT, 26), fill=(200, 210, 230))
    lines = [
        "・認識と判断は Jev（画像入力）: 物体の位置 = Box、対象かどうか・把持の成否 = Noul",
        "・物体名をハードコードしないクエリ形式で、開発に使っていない物体にも対応",
        "・動作は動作プリミティブ（pick / place）と IK。学習もデモデータも使わない",
        "・センサは実機でも取得できる構成のみ（固定の俯瞰カメラ、ハンドのリストカメラ）",
        "・MuJoCo（Franka Panda）+ dora-rs",
        "",
        "Demo 1  明示的な指示    Demo 2  開発に使っていない物体    Demo 3  外乱からの回復",
    ]
    for i, line in enumerate(lines):
        d.text((70, 260 + i * 46), line, font=font(JP_FONT_LIGHT, 25), fill=(220, 225, 235))
    d.text((70, 640), f"※ 冒頭と末尾のタイトル部分を除き {speed:g} 倍速で再生しています（{speed:g}x speed）",
           font=font(JP_FONT, 24), fill=(255, 220, 120))
    still(path, img, seconds, speed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--intro", type=float, default=7.0)
    ap.add_argument("--op", help="opening clip, played at 1x (scaled to 1280x720)")
    ap.add_argument("--ed", help="ending clip, played at 1x (scaled to 1280x720)")
    ap.add_argument("--cards", help="JSON file: per-clip title cards replacing the recorded ones")
    ap.add_argument("--card-seconds", type=float, default=4.0, help="card length after speed-up")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        intro(tdp / "intro.mp4", a.intro, a.speed)
        seq: list[Path] = [tdp / "intro.mp4"]
        cards = json.loads(Path(a.cards).read_text()) if a.cards else None
        if cards is not None and len(cards) != len(a.clips):
            raise SystemExit("--cards must have one entry per clip")
        for i, clip in enumerate(a.clips):
            if cards is None:
                seq.append(Path(clip))
                continue
            card(tdp / f"card{i}.mp4", cards[i]["title"], cards[i]["lines"], a.card_seconds, a.speed)
            trim(clip, tdp / f"clip{i}.mp4", RUN_CARD_S)
            seq += [tdp / f"card{i}.mp4", tdp / f"clip{i}.mp4"]
        lst = tdp / "list.txt"
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in seq))
        joined = tdp / "joined.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(lst), "-c", "copy", str(joined)], check=True)
        # Speed label as a PNG overlay (works without ffmpeg's drawtext/freetype).
        label = Image.new("RGBA", (190, 40), (0, 0, 0, 150))
        ImageDraw.Draw(label).text((12, 6), f"{a.speed:g}x speed", font=font(JP_FONT, 22),
                                   fill=(255, 255, 255, 255))
        label.save(tdp / "label.png")
        fc = f"[0:v]setpts=PTS/{a.speed},fps={FPS}[v];[v][1:v]overlay=W-w-14:H-h-12"
        main_mp4 = tdp / "main.mp4" if (a.op or a.ed) else Path(a.out)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-i",
                        str(tdp / "label.png"), "-filter_complex", fc, "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(main_mp4)],
                       check=True)
        if a.op or a.ed:
            parts = [p for p in (a.op, str(main_mp4), a.ed) if p]
            ins: list[str] = []
            for p in parts:
                ins += ["-i", p]
            norm = "".join(
                f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},setsar=1,format=yuv420p[v{i}];"
                for i in range(len(parts)))
            cat = "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[out]"
            subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", norm + cat,
                            "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                            a.out], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
