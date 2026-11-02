"""Step-1 probe, part 2: candidates from several spatially-disambiguated Box questions
(probe_ov2), each verified on a zoomed crop with Noul; pick the best-verified one.

uv run python scripts/probe_openvocab_verify.py outputs/probe_ov2 outputs/probe_ov3
"""

from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from probe_openvocab import INSTR, hit, scene_image

from tabletop.jev import JevClient
from tabletop.locate import crop_png, crop_window, to_full

VARS = ["single", "left", "top", "bottom"]
TIGHT = 0.12
WIDE = 0.3


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    jev = JevClient("jev", out / "jev_log.jsonl")
    from typesafe_sdk import Box, Noul

    logs = {}
    for line in (src / "jev_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        logs[r["tag"]] = r
    rows = json.loads((src / "probe.json").read_text())
    imgs: dict = {}
    jobs = []
    for r in rows:
        if not r["remaining_gt"]:
            continue
        seed, k, moved = r["seed"], r["instr"], r["moved"]
        key = (seed, tuple(moved))
        if key not in imgs:
            png, boxes = scene_image(seed, moved, {})
            imgs[key] = (Image.open(io.BytesIO(png)).convert("RGB"), boxes)
        a = logs[f"probe_s{seed}_i{k}_m{len(moved)}"]["answers"]
        cands: list[tuple] = []
        for v in VARS:
            b = a[f"t_{v}"]["box"]
            c = ((b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2)
            if all(abs(c[0] - d[0]) + abs(c[1] - d[1]) > 0.03 for d, _ in cands):
                cands.append((c, v))
        for c, v in cands:
            jobs.append((r, key, c, v))

    def verify(job):
        r, key, c, v = job
        img, boxes = imgs[key]
        l0, t0 = crop_window((c[0], c[1], c[0], c[1]), TIGHT)
        png, _ = crop_png(img, l0, t0, TIGHT)
        res = jev.ask(f"verify_s{r['seed']}_i{r['instr']}_m{len(r['moved'])}_{v}",
                      {"instruction": INSTR[r["instr"]][0],
                       "camera_info": "Small zoomed top-down crop of a floor workspace."},
                      {"centre": Noul(instructions="Is there an object at the centre of `camera`?"),
                       "box": Box(instructions="the single object nearest to the centre of `camera`"),
                       "wanted": Noul(instructions=(
                           "Is the object nearest to the centre of `camera` one that `instruction` "
                           "asks to put into the tray?"))},
                      {"camera": png})
        a = res.answers
        b = a["box"]["box"]
        fb = to_full((b["left"], b["top"], b["right"], b["bottom"]), l0, t0, TIGHT)
        c0 = c
        c = ((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2)
        return {"cand_obj": hit((*c0, *c0), boxes, list(boxes), 0.0), "bconf": a["box"].get("confidence"),"seed": r["seed"], "instr": r["instr"], "moved": r["moved"], "var": v, "c": c,
                "remaining_gt": r["remaining_gt"], "obj": hit((*c, *c), boxes, list(boxes), 0.0),
                "valid": hit((*c, *c), boxes, r["remaining_gt"], 0.0) is not None,
                "centre": a["centre"]["noul"], "wanted": a["wanted"]["noul"], "png": png}

    def track(x):
        img, boxes = imgs[(x["seed"], tuple(x["moved"]))]
        c = x["c"]
        l0, t0 = crop_window((c[0], c[1], c[0], c[1]), WIDE)
        png, _ = crop_png(img, l0, t0, WIDE)
        res = jev.ask(f"track_s{x['seed']}_i{x['instr']}_m{len(x['moved'])}",
                      {"camera_info": "Zoomed top-down crop of a floor workspace.",
                       "reference_info": "Close-up of the object the robot must pick."},
                      {"box": Box(instructions="the object in `camera` that looks like the object "
                                               "shown in `reference`")},
                      {"camera": png, "reference": x["png"]})
        b = res.answers["box"]["box"]
        fb = to_full((b["left"], b["top"], b["right"], b["bottom"]), l0, t0, WIDE)
        return hit(fb, boxes, list(boxes), 0.0), res.answers["box"].get("confidence")

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(verify, jobs))
    on = [x for x in res if x["obj"]]
    off = [x for x in res if not x["obj"]]
    print(f"candidates: {len(res)}, on an object {len(on)}")
    print(f"centre Noul: on-object p>=.5 {sum(x['centre'] >= .5 for x in on)}/{len(on)}, "
          f"off-object p<.5 {sum(x['centre'] < .5 for x in off)}/{len(off)}")
    print(f"wanted Noul correct on-object: {sum((x['wanted'] >= .5) == x['valid'] for x in on)}/{len(on)}")
    groups: dict = {}
    for x in res:
        groups.setdefault((x["seed"], x["instr"], tuple(x["moved"])), []).append(x)
    chosen = [max(g, key=lambda x: min(x["centre"], x["wanted"])) for g in groups.values()]
    print(f"pipeline (max min(centre,wanted)): {sum(x['valid'] for x in chosen)}/{len(chosen)}")
    for x in chosen:
        if not x["valid"]:
            print(" miss", x["seed"], x["instr"], x["moved"], x["var"], x["obj"],
                  [(y["var"], y["obj"], round(y["centre"], 2), round(y["wanted"], 2))
                   for y in groups[(x["seed"], x["instr"], tuple(x["moved"]))]])
    with ThreadPoolExecutor(8) as ex:
        tr = list(ex.map(track, [x for x in chosen if x["obj"]]))
    same = sum(t[0] == x["obj"] for t, x in zip(tr, [x for x in chosen if x["obj"]], strict=True))
    print(f"reference-tracking Box in {WIDE} crop lands on same object: {same}/{len(tr)}")
    for x in res:
        x.pop("png")
    (out / "verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
