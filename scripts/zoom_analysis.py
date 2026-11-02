"""Choose the adaptive-zoom threshold from development runs only.

Development runs use a fixed depth (two accepted zoom rounds) and log every round's
confidence and its grasp-point error vs. simulator truth. For each candidate threshold T
we replay the rule "stop after the first accepted round if its confidence >= T" and report
the resulting error distribution and number of zoom queries.

uv run python scripts/zoom_analysis.py outputs/dev_ov [more run roots...]
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def picks(roots: list[Path]) -> list[list[dict]]:
    out = []
    for root in roots:
        for m in sorted(root.glob("*/metrics.jsonl")):
            for line in m.read_text().splitlines():
                r = json.loads(line)
                if r["event"] == "grasp_point" and r["rounds"] and "err_mm" in r["rounds"][0]:
                    out.append(r["rounds"])
    return out


def replay(rounds: list[dict], T: float | None) -> tuple[float, int]:
    """Error and number of queries under the rule; the logs follow the same gate (0.3)."""
    for i, r in enumerate(rounds):
        if r["accepted"] and T is not None and r["conf"] >= T:
            return r["err_mm"], i + 1
    acc = [r for r in rounds if r["accepted"]]
    final = acc[-1] if acc else max(rounds, key=lambda r: r["conf"])
    return final["err_mm"], len(rounds)


def main() -> None:
    ps = picks([Path(p) for p in sys.argv[1:]])
    print(f"picks: {len(ps)}")
    first = [next((r for r in p if r["accepted"]), None) for p in ps]
    pairs = [(f["conf"], f["err_mm"]) for f in first if f]
    print("first accepted round: conf vs error (mm), sorted by conf")
    for c, e in sorted(pairs):
        print(f"  {c:.2f}  {e:5.1f}")
    rows = []
    for T in [None, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.3]:
        res = [replay(p, T) for p in ps]
        errs = [e for e, _ in res]
        q = [n for _, n in res]
        rows.append({"T": T, "median_mm": round(st.median(errs), 1), "p90_mm": round(sorted(errs)[int(0.9 * (len(errs) - 1))], 1),
                     "max_mm": round(max(errs), 1), "over_15mm": sum(e > 15 for e in errs),
                     "mean_queries": round(st.mean(q), 2)})
    print("rule: stop after first accepted round if conf >= T (None = always two rounds)")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
