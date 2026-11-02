"""Re-draw the banner subtitle of a recorded run.mp4 whose Japanese instruction was drawn as tofu
(the recorder used a Latin-only font before tabletop/video.py gained a CJK fallback).

uv run python scripts/fix_banner_text.py SRC.mp4 DST.mp4 "<instruction>"
"""
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, dst, text = sys.argv[1], sys.argv[2], sys.argv[3]
Y0, Y1 = 676, 708
BG = (14, 16, 22)
latin = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 19)
cjk = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 19)

def band(font):
    im = Image.new("RGB", (1280, 720), BG)
    ImageDraw.Draw(im).text((16, 624 + 56), text, font=font, fill=(200, 210, 230))
    return np.asarray(im)[Y0:Y1].astype(np.int16)

tofu, good = band(latin), band(cjk)
r = imageio.get_reader(src)
w = imageio.get_writer(dst, fps=25, codec="libx264", quality=9, macro_block_size=8)
n = fixed = 0
for f in r:  # pyright: ignore[reportGeneralTypeIssues]
    n += 1
    reg = f[Y0:Y1].astype(np.int16)
    if np.abs(reg - tofu).mean() < 6 and np.abs(tofu - BG).mean() > 0:
        f = f.copy()
        f[Y0:Y1] = good.astype(np.uint8)
        fixed += 1
    w.append_data(f)
w.close()
print(f"frames {n}, fixed {fixed}")
