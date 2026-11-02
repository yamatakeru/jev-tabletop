"""Thin wrapper around TypeSafe Jev (System One) for the tabletop task.

All perception and task-level decisions go through here:
  * overview(): which block next (Choice), where everything is (Box), was the last
    block placed (Noul)  -- one request, all questions evaluated in parallel
  * refine():   Box on a zoomed crop for a precise grasp point
  * held():     Noul on the front camera: is the gripper holding the block?

``mode="oracle"`` answers from simulator ground truth instead (debugging / baseline).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scene import COLORS

IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
OVERHEAD_INFO = (
    "Top-down camera photo of a floor workspace. The robot arm base is at the bottom "
    "centre of the image. A light-grey square tray with raised walls lies on the floor. "
    "Coloured cubes (red, green, blue, yellow) may lie on the floor or inside the tray."
)


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
        from typesafe_sdk import Image

        st = dict(state)
        for k, png in images.items():
            st[k] = Image(png)
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
            self.log.write(json.dumps(rec, default=str) + "\n")
            self.log.flush()

    # ------------------------------------------------------------ task level
    def overview(self, png: bytes, instruction: str, last_placed: str | None, gt: dict) -> dict:
        """Return {'next': str, 'next_probs': {..}, 'boxes': {name: (l,t,r,b)}, 'visible': {...},
        'placed_ok': float|None, 'latency_s': float}."""
        if self.mode == "oracle":
            return _oracle_overview(instruction, last_placed, gt)
        from typesafe_sdk import Box, Choice, Noul

        criteria = {c: f"The {c} cube" for c in COLORS}
        criteria["done"] = "No cube still needs to be moved: the instruction is already satisfied."
        qs: dict[str, Any] = {
            "next": Choice(
                instructions=(
                    "Following `instruction`, which cube should the robot pick up and put into "
                    "the tray next, as seen in `camera`? Only cubes that the instruction asks to "
                    "move AND that are still outside the tray are valid. Choose `done` if no "
                    "such cube remains."
                ),
                criteria=criteria,
            ),
            "box_tray": Box(instructions="the light-grey square tray in `camera`"),
        }
        for c in COLORS:
            qs[f"box_{c}"] = Box(instructions=f"the {c} cube in `camera`")
            qs[f"in_tray_{c}"] = Noul(instructions=f"In `camera`, is the {c} cube inside the tray?")
        r = self.ask("overview", {"instruction": instruction, "camera_info": OVERHEAD_INFO},
                     qs, {"camera": png})
        a = r.answers
        return {
            "next": a["next"]["choice"],
            "next_probs": a["next"]["probabilities"],
            "boxes": {k[4:]: _box(a[k]) for k in a if k.startswith("box_")},
            "box_conf": {k[4:]: a[k].get("confidence") for k in a if k.startswith("box_")},
            "in_tray": {c: a[f"in_tray_{c}"]["noul"] for c in COLORS},
            "latency_s": r.latency_s,
        }

    def refine(self, crop_png: bytes, color: str, gt_box: tuple | None) -> dict:
        if self.mode == "oracle":
            return {"box": gt_box, "conf": 1.0, "latency_s": 0.0}
        from typesafe_sdk import Box, Noul

        r = self.ask(
            f"refine_{color}",
            {"camera_info": "Zoomed top-down crop of the floor around a coloured cube."},
            {"box": Box(instructions=f"the {color} cube in `camera`"),
             "visible": Noul(instructions=f"Is a {color} cube visible in `camera`?")},
            {"camera": crop_png},
        )
        return {"box": _box(r.answers["box"]), "conf": r.answers["box"].get("confidence"),
                "visible": r.answers["visible"]["noul"], "latency_s": r.latency_s}

    def held(self, png: bytes, color: str, gt_held: bool) -> dict:
        if self.mode == "oracle":
            return {"held": 1.0 if gt_held else 0.0, "latency_s": 0.0}
        from typesafe_sdk import Noul

        r = self.ask(
            f"held_{color}",
            {"camera_info": "Wrist camera rigidly mounted on the robot hand, looking at the two "
                            "gripper fingers from above; the floor is behind them."},
            {"grasping": Noul(instructions=f"Is the robot gripper in `camera` grasping the {color} cube?"),
             "between": Noul(instructions=(
                 f"In `camera`, is there a {color} object between the two fingers of the gripper?"))},
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


def _oracle_overview(instruction: str, last_placed: str | None, gt: dict) -> dict:
    in_tray = gt["in_tray"]
    # Naive parser for the baseline: colours named before "leave" are targets.
    head = instruction.lower().split("leave")[0]
    wanted = [c for c in COLORS if c in head and not in_tray[c]]
    nxt = wanted[0] if wanted else "done"
    return {
        "next": nxt,
        "next_probs": {c: float(c == nxt) for c in [*COLORS, "done"]},
        "boxes": gt["boxes"],
        "box_conf": dict.fromkeys(gt["boxes"], 1.0),
        "in_tray": {c: float(v) for c, v in in_tray.items()},
        "latency_s": 0.0,
    }
