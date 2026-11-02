"""Batch experiments: robustness over layouts and instruction understanding.

uv run python scripts/experiments.py layouts|instructions|summary
Each run: outputs/exp/<name>/ (metrics.jsonl, jev_log.jsonl, final_state.json, config.json)
"""

from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "outputs" / "exp"
TRAY_HALF = 0.09
COLORS = ["red", "green", "blue", "yellow"]

BASE_INSTR = "Put the red, blue and green cubes into the tray. Leave the yellow cube alone."
LAYOUTS = [
    {"name": f"layout_seed{s}", "seed": s, "instruction": BASE_INSTR,
     "expected": ["red", "green", "blue"]}
    for s in (1, 2, 4, 5, 6)
]
INSTRUCTIONS = [
    {"name": "instr_japan_flag", "instruction": "Put the cubes that have a colour of the Japanese flag into the tray.", "expected": ["red"]},
    {"name": "instr_rgb_light", "instruction": "Put the cubes in the three primary colours of light into the tray.", "expected": ["red", "green", "blue"]},
    {"name": "instr_except_green", "instruction": "Put every cube except the green one into the tray.", "expected": ["red", "blue", "yellow"]},
    {"name": "instr_banana", "instruction": "Put the cube that is the colour of a ripe banana into the tray.", "expected": ["yellow"]},
    {"name": "instr_ja_explicit", "instruction": "赤と青のキューブをトレイに入れてください。", "expected": ["red", "blue"]},
    {"name": "instr_ja_traffic", "instruction": "信号機の色のキューブをトレイに入れて。", "expected": ["red", "green", "yellow"]},
]


def run(cfg: dict) -> None:
    out = EXP / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    env = {**os.environ, "RUN_DIR": str(out), "INSTRUCTION": cfg["instruction"], "NO_VIDEO": "1",
           "LAYOUT_SEED": str(cfg.get("seed", ""))}
    with open(out / "dora.log", "w") as log:
        subprocess.run(["dora", "run", "dataflow.yml", "--stop-after", "8m"], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=False)
    print(cfg["name"], "done", flush=True)


def evaluate(d: Path) -> dict:
    cfg = json.loads((d / "config.json").read_text())
    final = json.loads((d / "final_state.json").read_text())
    tx, ty, _ = final["tray"]
    in_tray = {c: abs(p[0] - tx) < TRAY_HALF and abs(p[1] - ty) < TRAY_HALF and p[2] < 0.08
               for c, p in final["blocks"].items()}
    exp = set(cfg["expected"])
    got = {c for c, v in in_tray.items() if v}
    m = [json.loads(line) for line in open(d / "metrics.jsonl")]
    jev = [json.loads(line) for line in open(d / "jev_log.jsonl")]
    grasp = [r for r in m if r["event"] == "grasp_point"]
    held = [r for r in m if r["event"] == "held_check"]
    first = next(r for r in m if r["event"] == "decision")
    return {
        "name": cfg["name"], "instruction": cfg["instruction"], "expected": sorted(exp),
        "in_tray": sorted(got), "success": got == exp,
        "first_choice": first["next"], "first_probs": first["probs"],
        "picks": len(grasp), "held_fail": sum(r["jev"] < 0.5 for r in held),
        "held_agree": sum((r["jev"] >= 0.5) == r["gt"] for r in held), "held_n": len(held),
        "coarse_mm": [round(r["coarse_err_mm"], 1) for r in grasp],
        "refined_mm": [round(r["refined_err_mm"], 1) for r in grasp],
        "jev_calls": len(jev),
        "jev_latency_median": round(st.median(r["latency_s"] for r in jev), 3) if jev else None,
        "input_tokens": sum(r["usage"].get("input_tokens") or 0 for r in jev),
        "wall_s": next((r["value"] for r in m if r["event"] == "wall_time_s"), None),
    }


def summary() -> None:
    rows = [evaluate(d) for d in sorted(EXP.iterdir()) if (d / "final_state.json").exists()]
    (EXP / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    ref = [v for r in rows for v in r["refined_mm"]]
    coa = [v for r in rows for v in r["coarse_mm"]]
    print(f"runs={len(rows)} success={sum(r['success'] for r in rows)}")
    if ref:
        print(f"grasp error coarse: median {st.median(coa):.1f} mm max {max(coa):.1f}; "
              f"refined: median {st.median(ref):.1f} mm max {max(ref):.1f}")


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "summary":
        summary()
    else:
        for cfg in {"layouts": LAYOUTS, "instructions": INSTRUCTIONS}[which]:
            run(cfg)
