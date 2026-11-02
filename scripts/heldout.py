"""Held-out evaluation, pre-registered before the open-vocabulary code was frozen.

Development used layout seeds 1-6 (3 in demo B), the four cube colours red/green/blue/yellow
and the instructions in scripts/experiments.py. Nothing below was seen during development:
new layout seeds (7+), new objects (a purple cube, an orange cylinder) and new phrasings
(English / Japanese). The expected result of every condition is fixed here, in advance.

Both versions run the same conditions:
  new  open-vocabulary version (tag openvocab-frozen), in this checkout
  old  closed-vocabulary v3 (tag closed-vocab-v3) in ../jev-tabletop-closed. Its Jev calls are
       unchanged; only tabletop/scene.py is replaced by the frozen one so that the new objects
       can be spawned (same layouts for both versions). The old prompts list four colour names,
       so conditions that need purple / orange / a cylinder are expected to fail there.

uv run python scripts/heldout.py run new|old      -> outputs/heldout_<version>/<name>/
uv run python scripts/heldout.py summary          -> outputs/heldout_summary.{json,md}
"""

from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT.parent / "jev-tabletop-closed"
TRAY_HALF = 0.09
BASE = "Put the red, blue and green cubes into the tray. Leave the yellow cube alone."
SIX = "red,green,blue,yellow,purple,orange:cylinder"

CONDITIONS: list[dict] = [
    # -- new layouts, development instruction and objects
    *({"name": f"layout_seed{s}", "group": "layout", "seed": s, "objects": "",
       "instruction": BASE, "expected": ["blue", "green", "red"]} for s in (7, 8, 9, 10, 11)),
    # -- new phrasings, development objects
    {"name": "para_clear_floor", "group": "phrasing", "seed": 15, "objects": "",
     "instruction": "Clear the floor: everything goes into the tray except the yellow block.",
     "expected": ["blue", "green", "red"]},
    {"name": "para_ja_only_blue", "group": "phrasing", "seed": 16, "objects": "",
     "instruction": "青いキューブだけを片付けてください。", "expected": ["blue"]},
    {"name": "para_warm_colours", "group": "phrasing", "seed": 17, "objects": "",
     "instruction": "Put the cubes in warm colours, like the colours of fire, into the tray.",
     "expected": ["red", "yellow"]},
    # -- new objects (+ new phrasings)
    {"name": "obj_ja_keep_purple", "group": "new_object", "seed": 12, "objects": SIX,
     "instruction": "紫の物は残して、それ以外をすべてトレイに片付けて。",
     "expected": ["blue", "green", "orange_cylinder", "red", "yellow"]},
    {"name": "obj_purple_and_orange", "group": "new_object", "seed": 13, "objects": SIX,
     "instruction": "Put the purple cube and the orange cylinder into the tray.",
     "expected": ["orange_cylinder", "purple"]},
    {"name": "obj_round_one", "group": "new_object", "seed": 14,
     "objects": "red,green,blue,yellow,orange:cylinder",
     "instruction": "Tidy away the round object.", "expected": ["orange_cylinder"]},
]


def out_dir(version: str) -> Path:
    return ROOT / "outputs" / f"heldout_{version}"


def run(version: str) -> None:
    cwd = ROOT if version == "new" else OLD_ROOT
    for cfg in CONDITIONS:
        out = out_dir(version) / cfg["name"]
        if (out / "final_state.json").exists():
            print(cfg["name"], "exists, skipped", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({**cfg, "version": version},
                                                    ensure_ascii=False, indent=2))
        env = {**os.environ, "JEV_MODEL": "jev-1.14.0", "RUN_DIR": str(out), "NO_VIDEO": "1",
               "INSTRUCTION": cfg["instruction"], "LAYOUT_SEED": str(cfg["seed"]),
               "OBJECTS": cfg["objects"]}
        with open(out / "dora.log", "w") as log:
            subprocess.run(["dora", "run", "dataflow.yml", "--stop-after", "8m"], cwd=cwd,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        print(cfg["name"], "done", flush=True)


def _lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def evaluate(d: Path) -> dict:
    cfg = json.loads((d / "config.json").read_text())
    row: dict = {"name": cfg["name"], "group": cfg["group"], "expected": sorted(cfg["expected"])}
    if not (d / "final_state.json").exists():
        return {**row, "success": False, "error": "no final state (crashed)"}
    final = json.loads((d / "final_state.json").read_text())
    tx, ty, _ = final["tray"]
    got = sorted(c for c, p in final["blocks"].items()
                 if abs(p[0] - tx) < TRAY_HALF and abs(p[1] - ty) < TRAY_HALF and p[2] < 0.08)
    m = _lines(d / "metrics.jsonl")
    jev = _lines(d / "jev_log.jsonl")
    grasp = [r for r in m if r["event"] == "grasp_point"]
    held = [r for r in m if r["event"] == "held_check"]
    first_try: dict[str, bool] = {}
    for r in held:
        first_try.setdefault(r["color"], r["gt"])
    zoom = [len(r["rounds"]) for r in grasp]
    return {
        **row, "in_tray": got, "success": got == row["expected"],
        "picks": len(grasp), "first_grasp_ok": sum(first_try.values()), "objects_tried": len(first_try),
        "held_agree": sum((r["jev"] >= 0.5) == r["gt"] for r in held), "held_n": len(held),
        "held_p_pos": [round(r["jev"], 2) for r in held if r["gt"]],
        "held_p_neg": [round(r["jev"], 2) for r in held if not r["gt"]],
        "n_candidates": [len(r.get("candidates", [])) for r in m if r["event"] == "decision"],
        "refined_mm": [round(r["refined_err_mm"], 1) for r in grasp],
        "zoom_queries": zoom,
        "jev_calls": len(jev), "jev_latency_sum_s": round(sum(r["latency_s"] for r in jev), 2),
        "wall_s": next((round(r["value"], 1) for r in m if r["event"] == "wall_time_s"), None),
    }


def _mm(v: list[float]) -> str:
    return f"{min(v):.2f} / {max(v):.2f} (n={len(v)})" if v else "- (n=0)"


def summary() -> None:
    res = {}
    for v in ("new", "old"):
        d = out_dir(v)
        res[v] = [evaluate(d / c["name"]) for c in CONDITIONS if (d / c["name"] / "config.json").exists()]
    (ROOT / "outputs" / "heldout_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    md = ["# Held-out evaluation (pre-registered in scripts/heldout.py)", "",
          "| condition | expected | new: in tray | new | old: in tray | old |", "|---|---|---|---|---|---|"]
    old = {r["name"]: r for r in res.get("old", [])}
    for r in res.get("new", []):
        o = old.get(r["name"], {})
        md.append(f"| {r['name']} | {', '.join(r['expected'])} | {', '.join(r.get('in_tray', []))} | "
                  f"{'OK' if r['success'] else 'FAIL'} | {', '.join(o.get('in_tray', [])) or o.get('error', '')} | "
                  f"{'OK' if o.get('success') else ('FAIL' if o else '-')} |")
    md += ["", "| metric | new | old |", "|---|---|---|"]

    def agg(rows: list[dict]) -> dict:
        ok = [r for r in rows if "picks" in r]
        ref = [v for r in ok for v in r["refined_mm"]]
        out = {
            "success": f"{sum(r['success'] for r in rows)}/{len(rows)}",
            "success (layout / phrasing / new object)": " / ".join(
                f"{sum(r['success'] for r in rows if r['group'] == g)}/{sum(r['group'] == g for r in rows)}"
                for g in ("layout", "phrasing", "new_object")),
            "first-try grasp success": f"{sum(r['first_grasp_ok'] for r in ok)}/{sum(r['objects_tried'] for r in ok)}",
            "held check agrees with truth": f"{sum(r['held_agree'] for r in ok)}/{sum(r['held_n'] for r in ok)}",
            "held p when holding (min / max)": _mm([v for r in ok for v in r["held_p_pos"]]),
            "held p when not holding (min / max)": _mm([v for r in ok for v in r["held_p_neg"]]),
            "held misjudged": str(sum(r["held_n"] - r["held_agree"] for r in ok)),
            "grasp error after zoom, median / max (mm)":
                f"{st.median(ref):.1f} / {max(ref):.1f}" if ref else "-",
            "zoom queries per pick, mean": f"{st.mean(v for r in ok for v in r['zoom_queries']):.2f}"
                if any(r["zoom_queries"] for r in ok) else "-",
            "Jev calls per pick": f"{sum(r['jev_calls'] for r in ok) / max(1, sum(r['picks'] for r in ok)):.1f}",
            "Jev latency per pick (s, sum of calls)":
                f"{sum(r['jev_latency_sum_s'] for r in ok) / max(1, sum(r['picks'] for r in ok)):.2f}",
            "wall time per episode, median (s)":
                f"{st.median(r['wall_s'] for r in ok if r['wall_s']):.0f}" if ok else "-",
        }
        return out

    a_new, a_old = agg(res.get("new", [])), agg(res.get("old", []))
    for k in a_new:
        md.append(f"| {k} | {a_new[k]} | {a_old.get(k, '-')} |")
    (ROOT / "outputs" / "heldout_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        summary()
