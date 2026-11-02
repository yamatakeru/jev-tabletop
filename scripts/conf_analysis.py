"""Box confidence vs. grasp-point error (report section on the confidence gate).

For every pick (metrics.jsonl "grasp_point"): the minimum Box confidence over its zoom
queries and the grasp-point error after zoom (vs. simulator truth, evaluation only).

  (a) table: picks with min confidence < 0.3 / >= 0.3 -> count, median / max error
  (b) CSV of points for figures: one row per zoom query (version, run, step, object,
      round, conf, accepted, min_conf, refined_err_mm, round_err_mm if logged)

uv run python scripts/conf_analysis.py [--until HH:MM:SS] [--csv out.csv] LABEL=DIR_OR_GLOB ...
  DIR_OR_GLOB: a run directory, a directory of runs, or a glob of run directories.
  --until: only runs whose metrics.jsonl was last written at or before this local time
           (reproduces a snapshot of the logs as they were when a decision was made).

The report's 97 estimates are all those logged before the confidence gate was added (tag
exp-v3): runs of the exp-v1 / exp-v2 stages plus trial, demo and interactive runs, e.g.
  uv run python scripts/conf_analysis.py --until 04:35:34 \\
    v1=<exp-v1 runs> v2=<exp-v2 runs> trials='outputs/trial_*' demos=<demo runs>
The resulting table is saved as results/conf_analysis/report96.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
import time
from pathlib import Path

GATE = 0.3


def run_dirs(spec: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(glob.glob(spec)) or [spec]:
        d = Path(p)
        if (d / "metrics.jsonl").exists():
            out.append(d)
        elif d.is_dir():
            out += sorted(x.parent for x in d.glob("*/metrics.jsonl"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", help="HH:MM[:SS] local time cutoff on metrics.jsonl mtime")
    ap.add_argument("--csv", type=Path, help="write per-query points here")
    ap.add_argument("sets", nargs="+", help="LABEL=DIR_OR_GLOB")
    a = ap.parse_args()
    picks: list[dict] = []
    for item in a.sets:
        label, _, spec = item.partition("=")
        for d in run_dirs(spec):
            m = d / "metrics.jsonl"
            mt = time.localtime(m.stat().st_mtime)
            if a.until and time.strftime("%H:%M:%S", mt)[: len(a.until)] > a.until:
                continue
            for line in m.read_text().splitlines():
                r = json.loads(line)
                if r["event"] != "grasp_point" or not r["rounds"]:
                    continue
                picks.append({"version": label, "run": str(d), "step": r["step"],
                              "object": r["color"], "rounds": r["rounds"],
                              "min_conf": min(x["conf"] or 0.0 for x in r["rounds"]),
                              "refined_err_mm": r["refined_err_mm"]})
    print(f"picks: {len(picks)}")
    print("| min Box confidence while zooming | picks | error after zoom, median / max |")
    print("|---|---|---|")
    for name, sel in ((f"< {GATE}", lambda p: p["min_conf"] < GATE),
                      (f">= {GATE}", lambda p: p["min_conf"] >= GATE)):
        e = [p["refined_err_mm"] for p in picks if sel(p)]
        cell = f"{st.median(e):.1f} mm / {max(e):.0f} mm" if e else "-"
        print(f"| {name} | {len(e)} | {cell} |")
    by_v: dict[str, list[float]] = {}
    for p in picks:
        by_v.setdefault(p["version"], []).append(p["refined_err_mm"])
    for v, e in by_v.items():
        print(f"  {v}: {len(e)} picks, error median {st.median(e):.1f} / max {max(e):.1f} mm")
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["version", "run", "step", "object", "round", "conf", "accepted",
                        "min_conf", "refined_err_mm", "round_err_mm"])
            for p in picks:
                for x in p["rounds"]:
                    w.writerow([p["version"], p["run"], p["step"], p["object"], x["round"],
                                x["conf"], x.get("accepted", ""), p["min_conf"],
                                round(p["refined_err_mm"], 2),
                                round(x["err_mm"], 2) if "err_mm" in x else ""])
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
