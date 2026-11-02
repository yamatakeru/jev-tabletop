# pyright: reportAttributeAccessIssue=false
"""dora node: skill executor. Expands a skill into motion segments and streams joint targets.

inputs
  robot_state  float64[13] from sim (sim_time, q[7], fingers[2], tcp[3])
  skill        JSON string {"name": "pick"|"place"|"park"|"open", "xy": [x, y]}
outputs
  joint_cmd    float64[8] (7 joint targets + gripper 0..255)
  skill_done   JSON string {"name": ..., "ok": bool}
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from dora import Node

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabletop.ik import IKSolver  # noqa: E402
from tabletop.scene import BLOCK_HALF, PARK_Q, R_DOWN, load_model  # noqa: E402

OPEN, CLOSED = 255.0, 0.0
HOVER_Z = 0.20
GRASP_Z = BLOCK_HALF - 0.004  # fingertip centre slightly below the block centre
PLACE_Z = 0.10
CART_SPEED = 0.18  # m/s
JOINT_SPEED = 1.2  # rad/s


@dataclass
class Segment:
    kind: str  # "joint" | "cart" | "grip"
    target: Any = None
    duration: float = 0.0
    q0: Any = None
    p0: Any = None
    t0: float = -1.0
    extra: dict = field(default_factory=dict)


def smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class Executor:
    def __init__(self) -> None:
        model, _ = load_model()
        self.ik = IKSolver(model)
        self.queue: list[Segment] = []
        self.q_des = np.array(PARK_Q, dtype=float)
        self.grip = OPEN
        self.current: Segment | None = None
        self.skill_name: str | None = None

    def plan(self, skill: dict) -> None:
        name = skill["name"]
        self.skill_name = name
        if name in ("pick", "place"):
            x, y = skill["xy"]
            low = GRASP_Z if name == "pick" else PLACE_Z
            above = np.array([x, y, HOVER_Z])
            q_above = self.ik.solve_global(self.q_des, above, R_DOWN)
            segs = []
            if name == "pick":
                segs.append(Segment("grip", OPEN, 0.3))
            segs += [
                Segment("joint", q_above),
                Segment("cart", np.array([x, y, low])),
                Segment("grip", CLOSED if name == "pick" else OPEN, 0.8 if name == "pick" else 0.5),
                Segment("cart", np.array([x, y, HOVER_Z + 0.05])),
            ]
            self.queue = segs
        elif name == "park":
            self.queue = [Segment("joint", np.array(PARK_Q, dtype=float))]
        elif name == "open":
            self.queue = [Segment("grip", OPEN, 0.5)]
        else:
            raise ValueError(f"unknown skill {name}")

    def step(self, t: float, q: np.ndarray, tcp: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """Return (joint_cmd or None, finished_flag)."""
        if self.current is None:
            if not self.queue:
                return None, False
            seg = self.queue.pop(0)
            seg.t0 = t
            seg.q0 = self.q_des.copy()
            seg.p0 = tcp.copy()
            if seg.kind == "joint":
                seg.duration = max(1.0, float(np.max(np.abs(seg.target - seg.q0))) / JOINT_SPEED)
            elif seg.kind == "cart":
                seg.duration = max(0.6, float(np.linalg.norm(seg.target - seg.p0)) / CART_SPEED)
            self.current = seg
        seg = self.current
        s = smooth((t - seg.t0) / seg.duration)
        if seg.kind == "joint":
            self.q_des = seg.q0 + (seg.target - seg.q0) * s
        elif seg.kind == "cart":
            p = seg.p0 + (seg.target - seg.p0) * s
            self.q_des = self.ik.solve(self.q_des, p, R_DOWN, iters=20)
        elif seg.kind == "grip":
            self.grip = float(seg.target)
        settle = 0.4 if seg.kind != "grip" else 0.0
        done = t - seg.t0 >= seg.duration + settle
        if done:
            self.current = None
        cmd = np.concatenate([self.q_des, [self.grip]])
        finished = done and not self.queue
        return cmd, finished


def main() -> None:
    node = Node()
    ex = Executor()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "skill":
            skill = json.loads(event["value"][0].as_py())
            ex.plan(skill)
        elif event["id"] == "robot_state":
            st = event["value"].to_numpy()
            t, q, tcp = st[0], st[1:8], st[10:13]
            cmd, finished = ex.step(t, q, tcp)
            if cmd is not None:
                node.send_output("joint_cmd", pa.array(cmd))
            if finished:
                node.send_output(
                    "skill_done", pa.array([json.dumps({"name": ex.skill_name, "ok": True})])
                )


if __name__ == "__main__":
    main()
