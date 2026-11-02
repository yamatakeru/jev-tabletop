# pyright: reportAttributeAccessIssue=false
"""Step-1 probe: can Jev pick an instruction-relevant object without a closed colour list?

Renders overhead images of the development layouts (seeds 1-6) directly with MuJoCo, in the
initial state and with some target cubes already moved into the tray, and asks:
  Box  target    "an object the instruction asks to put in the tray and still outside it"
  Noul remaining "is such an object still outside the tray?"
  Box  outside   "an object outside the tray" (instruction-independent fallback)
Scored against simulator ground truth (evaluation only).

uv run python scripts/probe_openvocab.py [out_dir]
"""

from __future__ import annotations

import io
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tabletop.jev import JevClient
from tabletop.scene import (
    BLOCK_HALF,
    load_model,
    plane_to_pixel,
    random_layout,
)

BASE = "Put the red, blue and green cubes into the tray. Leave the yellow cube alone."
INSTR = [
    (BASE, ["red", "blue", "green"]),
    ("Put the cubes that have a colour of the Japanese flag into the tray.", ["red"]),
    ("Put the cubes in the three primary colours of light into the tray.", ["red", "green", "blue"]),
    ("Put every cube except the green one into the tray.", ["red", "blue", "yellow"]),
    ("Put the cube that is the colour of a ripe banana into the tray.", ["yellow"]),
    ("赤と青のキューブをトレイに入れてください。", ["red", "blue"]),
    ("信号機の色のキューブをトレイに入れて。", ["red", "green", "yellow"]),
]
INFO = ("Top-down camera photo of a floor workspace. The robot arm base is at the bottom centre "
        "of the image. A light-grey square tray with raised walls lies on the floor. Small "
        "objects may lie on the floor or inside the tray.")
VARIANTS = {
    "plain": "one {t}.",
    "single": "exactly one single {t}. Box only that one object, not a group of objects.",
    "left": "{t}; if there are several, the one nearest to the left edge of the image.",
    "top": "{t}; if there are several, the one nearest to the top edge of the image.",
    "neartray": "{t}; if there are several, the one nearest to the tray.",
    "bottom": "{t}; if there are several, the one nearest to the bottom edge of the image.",
}
PLACE = [(-0.035, 0.035), (0.035, -0.035), (0.035, 0.035), (-0.035, -0.035)]


def scene_image(seed: int, moved: list[str], renderer_cache: dict):
    lay = random_layout(seed)
    model, data = load_model(lay)
    tx, ty = lay.tray
    for i, c in enumerate(moved):
        jid = model.joint(f"block_{c}_joint").id
        adr = model.jnt_qposadr[jid]
        data.qpos[adr:adr + 7] = [tx + PLACE[i][0], ty + PLACE[i][1], BLOCK_HALF + 0.006, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, 768, 768)
    r.update_scene(data, "overhead")
    img = r.render()
    r.close()
    boxes = {}
    for c in lay.blocks:
        p = data.xpos[model.body(f"block_{c}").id]
        uv = np.array([plane_to_pixel(model, data, p + np.array(s) * BLOCK_HALF)
                       for s in itertools.product([-1, 1], repeat=3)])
        boxes[c] = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue(), boxes


def hit(box, boxes: dict, names, margin=0.01) -> str | None:
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for n in names:
        b = boxes[n]
        if b[0] - margin <= u <= b[2] + margin and b[1] - margin <= v <= b[3] + margin:
            return n
    return None


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs" / "probe_ov")
    out.mkdir(parents=True, exist_ok=True)
    jev = JevClient("jev", out / "jev_log.jsonl")
    from typesafe_sdk import Box, Noul

    cases = []
    for seed in (1, 2, 3, 4, 5, 6):
        for k, (instr, exp) in enumerate(INSTR):
            # states: nothing moved, half moved, all expected moved (done)
            states = sorted({0, len(exp) // 2 if len(exp) > 1 else 0, len(exp)})
            for n in states:
                cases.append((seed, k, instr, exp, exp[:n]))
    imgs = {}
    for seed, _k, _i, _e, moved in cases:
        key = (seed, tuple(moved))
        if key not in imgs:
            imgs[key] = scene_image(seed, moved, {})

    def ask(case):
        seed, k, instr, exp, moved = case
        png, boxes = imgs[(seed, tuple(moved))]
        tgt = ("an object that `instruction` asks to put into the tray and that is still "
               "outside the tray")
        qs = {f"t_{k}": Box(instructions=f"In `camera`, {q.format(t=tgt)}") for k, q in VARIANTS.items()}
        qs["remaining"] = Noul(instructions=(
            "In `camera`, is there still at least one object outside the tray that "
            "`instruction` asks to put into the tray?"))
        r = jev.ask(f"probe_s{seed}_i{k}_m{len(moved)}", {"instruction": instr, "camera_info": INFO},
                    qs, {"camera": png})
        a = r.answers
        remaining = [c for c in exp if c not in moved]
        row = {"seed": seed, "instr": k, "moved": moved, "remaining_gt": remaining,
               "remaining_p": a["remaining"]["noul"], "latency": r.latency_s}
        for v in VARIANTS:
            bb = a[f"t_{v}"]["box"]
            bb = (bb["left"], bb["top"], bb["right"], bb["bottom"])
            row[v] = {"hit": hit(bb, boxes, remaining), "any": hit(bb, boxes, list(boxes)),
                      "conf": a[f"t_{v}"].get("confidence"), "w": max(bb[2] - bb[0], bb[3] - bb[1])}
        return row

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(ask, cases))
    (out / "probe.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    act = [r for r in rows if r["remaining_gt"]]
    done = [r for r in rows if not r["remaining_gt"]]
    multi = [r for r in act if len(r["remaining_gt"]) > 1]
    for v in VARIANTS:
        h = sum(r[v]["hit"] is not None for r in act)
        hm = sum(r[v]["hit"] is not None for r in multi)
        print(f"{v:10s} hit {h}/{len(act)}  (multi-valid {hm}/{len(multi)})")
    best = sum(max((r[v] for v in VARIANTS), key=lambda x: x["conf"] or 0)["hit"] is not None
               for r in act)
    print(f"most-confident variant hit {best}/{len(act)}")
    rem_ok = sum(r["remaining_p"] >= 0.5 for r in act) + sum(r["remaining_p"] < 0.5 for r in done)
    print(f"remaining Noul correct: {rem_ok}/{len(rows)} (active {sum(r['remaining_p'] >= 0.5 for r in act)}/{len(act)}, "
          f"done {sum(r['remaining_p'] < 0.5 for r in done)}/{len(done)})")

if __name__ == "__main__":
    main()
