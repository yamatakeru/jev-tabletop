"""Thin wrapper around TypeSafe Jev (System One) for the tabletop task.

Open-vocabulary: no object names or colours appear in the questions. The instruction text is
passed as state and every question refers to "an object that `instruction` asks to put into
the tray"; a chosen object is then tracked by its *appearance* (a close-up crop passed as a
`reference` image), not by a name.

  * overview(): one request with several Box questions for a target object (spatial
    tie-breakers: a plain Box over several matching objects tends to land between them),
    the tray Box, "is any target left?" (Noul) and "is the last object in the tray?" (Noul);
    then each candidate is verified on a tight crop (Noul x2, requests in parallel).
  * refine():  Box on a zoomed crop: "the object that looks like `reference`"
  * held():    Noul on the wrist camera: is the gripper holding an object?

``mode="oracle"`` answers from simulator ground truth instead (debugging / baseline).
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .locate import crop_png, crop_window, to_full
from .scene import OBJECTS

IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
OVERHEAD_INFO = (
    "Top-down camera photo of a floor workspace. The robot arm base is at the bottom "
    "centre of the image. A light-grey square tray with raised walls lies on the floor. "
    "Small objects may lie on the floor or inside the tray."
)
REFERENCE_INFO = "Close-up (top-down) of the object the robot has to pick."
TARGET = ("an object that `instruction` asks to put into the tray and that is still outside "
          "the tray")
# Spatially disambiguated variants of the same question. Each yields one candidate; several
# variants are needed because one Box over several matching objects often lands between them
# (probe on the development layouts: plain 26/72 hits, any of these four 72/72).
VARIANTS = {
    "single": f"In `camera`, exactly one single {TARGET}. Box only that one object, not a group "
              "of objects.",
    "left": f"In `camera`, {TARGET}; if there are several, the one nearest to the left edge of "
            "the image.",
    "top": f"In `camera`, {TARGET}; if there are several, the one nearest to the top edge of the "
           "image.",
    "bottom": f"In `camera`, {TARGET}; if there are several, the one nearest to the bottom edge "
              "of the image.",
}
VERIFY_CROP = 0.08  # fraction of the overhead image (~9 cm on the floor) around a candidate
RECENTRE_CROP = 0.16  # crop for re-asking Box when no candidate is centred on an object
DEDUP = 0.03  # candidates closer than this (L1, image fraction) are the same
SELECT_MIN = 0.5  # min(centre, wanted) of the best candidate; below -> nothing to do
REMAINING_MIN = 0.5  # "is anything left to move?" Noul
BOX_HALF = 0.02  # initial box around a verified candidate point (~ one object)


@dataclass
class JevResult:
    answers: dict[str, Any]
    latency_s: float
    model: str
    usage: dict[str, Any]


class JevClient:
    def __init__(self, mode: str = "jev", log_path: Path | None = None):
        self.mode = mode
        self.log = open(log_path, "a") if log_path else None  # noqa: SIM115
        self._lock = threading.Lock()
        self.client = None
        if mode == "jev":
            from dotenv import load_dotenv
            from typesafe_sdk import TypeSafeClient

            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
            if not (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")):
                raise RuntimeError("TYPESAFE_API_KEY not set (put it in .env)")
            key = os.environ.get("TYPESAFE_API_KEY") or os.environ["JEV_API_KEY"]
            self.client = TypeSafeClient(api_key=key, model=os.environ.get("JEV_MODEL", "jev-latest"),
                                         timeout=60)

    # ------------------------------------------------------------- low level
    def ask(self, tag: str, state: dict, questions: dict, images: dict[str, bytes]) -> JevResult:
        from typesafe_sdk import Image as JevImage

        st = dict(state)
        for k, png in images.items():
            st[k] = JevImage(png)
        t0 = time.time()
        assert self.client is not None
        resp = self.client.system_one(state=st, questions=questions, extra_headers=IMAGE_HEADERS)
        dt = time.time() - t0
        answers = {k: v.model_dump() for k, v in resp.answers.items()}
        usage = resp.usage.model_dump()
        self._log({"tag": tag, "latency_s": round(dt, 3), "model": resp.model,
                   "state_text": dict(state),
                   "questions": {k: _q_repr(q) for k, q in questions.items()},
                   "answers": answers, "usage": usage})
        return JevResult(answers, dt, resp.model, usage)

    def _log(self, rec: dict) -> None:
        if self.log:
            with self._lock:
                self.log.write(json.dumps(rec, default=str) + "\n")
                self.log.flush()

    # ------------------------------------------------------------ task level
    def overview(self, png: bytes, instruction: str, last_ref: bytes | None, gt: dict) -> dict:
        """Pick the next object to move.

        Returns {'target': box|None (None = done), 'ref_png': close-up of the target,
        'tray': box, 'remaining': p, 'placed_ok': p|None, 'candidates': [...],
        'latency_s': wall time of all requests, 'n_queries': int}."""
        if self.mode == "oracle":
            return _oracle_overview(instruction, gt)
        from typesafe_sdk import Box, Noul

        qs: dict[str, Any] = {f"t_{k}": Box(instructions=v) for k, v in VARIANTS.items()}
        qs["tray"] = Box(instructions="the light-grey square tray in `camera`")
        qs["remaining"] = Noul(instructions=(
            "In `camera`, is there still at least one object outside the tray that "
            "`instruction` asks to put into the tray?"))
        images = {"camera": png}
        state = {"instruction": instruction, "camera_info": OVERHEAD_INFO}
        if last_ref is not None:
            images["reference"] = last_ref
            state["reference_info"] = "Close-up of the object the robot just put down."
            qs["placed"] = Noul(instructions=(
                "In `camera`, is the object shown in `reference` inside the tray?"))
        t0 = time.time()
        r = self.ask("overview", state, qs, images)
        a = r.answers
        out: dict[str, Any] = {
            "tray": _box(a["tray"]), "remaining": a["remaining"]["noul"],
            "placed_ok": a["placed"]["noul"] if "placed" in a else None,
            "candidates": [], "target": None, "ref_png": None, "n_queries": 1,
        }
        cands: list[dict] = []
        for k in VARIANTS:
            b = _box(a[f"t_{k}"])
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if all(abs(c[0] - d["c"][0]) + abs(c[1] - d["c"][1]) > DEDUP for d in cands):
                cands.append({"var": k, "c": c, "box": b, "conf": a[f"t_{k}"].get("confidence")})
        if out["remaining"] >= REMAINING_MIN:
            img = Image.open(io.BytesIO(png)).convert("RGB")
            with ThreadPoolExecutor(len(cands)) as ex:
                res = list(ex.map(lambda cd: self._verify(img, cd, instruction), cands))
            out["n_queries"] += len(cands)
            best = max(res, key=lambda cd: cd["score"])
            seed = max(res, key=lambda cd: cd["wanted"])
            if best["score"] < SELECT_MIN and seed["wanted"] >= SELECT_MIN:
                # Something wanted is near a candidate but off-centre: re-ask Box on a zoomed
                # crop around it and verify again (development: 2 of 49 decisions needed this).
                res.append(self._verify(img, self._recentre(img, seed, instruction), instruction))
                out["n_queries"] += 2
                best = max(res, key=lambda cd: cd["score"])
            if best["score"] >= SELECT_MIN:
                cu, cv = best["c"]
                out["target"] = (cu - BOX_HALF, cv - BOX_HALF, cu + BOX_HALF, cv + BOX_HALF)
                out["ref_png"] = best.pop("png")
            for cd in res:
                cd.pop("png", None)
            cands = res
        out["candidates"] = cands
        out["latency_s"] = time.time() - t0
        return out

    def _recentre(self, img: Image.Image, cd: dict, instruction: str) -> dict:
        from typesafe_sdk import Box

        c = cd["c"]
        l0, t0 = crop_window((c[0], c[1], c[0], c[1]), RECENTRE_CROP)
        png, _ = crop_png(img, l0, t0, RECENTRE_CROP)
        r = self.ask("recentre", {"instruction": instruction,
                                  "camera_info": "Zoomed top-down crop of a floor workspace."},
                     {"box": Box(instructions=VARIANTS["single"])}, {"camera": png})
        b = to_full(_box(r.answers["box"]), l0, t0, RECENTRE_CROP)
        return {"var": "recentre", "c": ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), "box": b,
                "conf": r.answers["box"].get("confidence")}

    def _verify(self, img: Image.Image, cd: dict, instruction: str) -> dict:
        from typesafe_sdk import Noul

        c = cd["c"]
        l0, t0 = crop_window((c[0], c[1], c[0], c[1]), VERIFY_CROP)
        png, _ = crop_png(img, l0, t0, VERIFY_CROP)
        r = self.ask(f"verify_{cd['var']}",
                     {"instruction": instruction,
                      "camera_info": "Small zoomed top-down crop of a floor workspace."},
                     {"centre": Noul(instructions="Is there an object at the centre of `camera`?"),
                      "wanted": Noul(instructions=(
                          "Is the object at the centre of `camera` one that `instruction` asks "
                          "to put into the tray?"))},
                     {"camera": png})
        ce, wa = r.answers["centre"]["noul"], r.answers["wanted"]["noul"]
        return {**cd, "centre": ce, "wanted": wa, "score": min(ce, wa), "png": png,
                "latency_s": r.latency_s}

    def refine(self, crop: bytes, ref_png: bytes | None, gt_box: tuple | None) -> dict:
        if self.mode == "oracle":
            return {"box": gt_box, "conf": 1.0, "latency_s": 0.0}
        from typesafe_sdk import Box, Noul

        assert ref_png is not None
        r = self.ask(
            "refine",
            {"camera_info": "Zoomed top-down crop of a floor workspace.",
             "reference_info": REFERENCE_INFO},
            {"box": Box(instructions="the object in `camera` that looks like the object shown in "
                                     "`reference`"),
             "visible": Noul(instructions=(
                 "Is the object shown in `reference` visible in `camera`?"))},
            {"camera": crop, "reference": ref_png},
        )
        return {"box": _box(r.answers["box"]), "conf": r.answers["box"].get("confidence"),
                "visible": r.answers["visible"]["noul"], "latency_s": r.latency_s}

    def held(self, png: bytes, gt_held: bool) -> dict:
        """Generic grasp check (no reference image: only one object can be in the hand, and
        on development wrist images a reference close-up made Jev under-confident)."""
        if self.mode == "oracle":
            return {"held": 1.0 if gt_held else 0.0, "latency_s": 0.0}
        from typesafe_sdk import Noul

        r = self.ask(
            "held",
            {"camera_info": "Wrist camera rigidly mounted on the robot hand, looking at the two "
                            "gripper fingers from above; the floor is behind them."},
            {"grasping": Noul(instructions="Is the robot gripper in `camera` grasping an object?"),
             "between": Noul(instructions=(
                 "In `camera`, is there an object between the two fingers of the gripper?"))},
            {"camera": png},
        )
        a = r.answers
        p = (a["grasping"]["noul"] + a["between"]["noul"]) / 2
        return {"held": p, "grasping": a["grasping"]["noul"], "between": a["between"]["noul"],
                "latency_s": r.latency_s}


def _box(ans: dict) -> tuple[float, float, float, float]:
    b = ans["box"]
    return (b["left"], b["top"], b["right"], b["bottom"])


def _q_repr(q: Any) -> Any:
    try:
        return q.model_dump()
    except AttributeError:
        return str(q)


# ------------------------------------------------------------------ oracle


def _oracle_overview(instruction: str, gt: dict) -> dict:
    """Baseline with simulator truth and a naive parser: colours named before "leave"."""
    in_tray = gt["in_tray"]
    head = instruction.lower().split("leave")[0]
    wanted = [n for n, (colour, _) in OBJECTS.items() if colour in head and not in_tray[n]]
    return {
        "target": gt["boxes"][wanted[0]] if wanted else None,
        "ref_png": None, "tray": gt["boxes"]["tray"],
        "remaining": float(bool(wanted)), "placed_ok": None, "candidates": [],
        "latency_s": 0.0, "n_queries": 0,
    }
