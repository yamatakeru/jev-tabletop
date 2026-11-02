# pyright: reportAttributeAccessIssue=false
"""dora node: the task loop. Jev looks at camera images and decides; code turns decisions
into skills. No training, no demonstrations.

inputs   camera_image (from sim), skill_done (from controller)
outputs  render_req, skill, overlay
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from dora import Node
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabletop.jev import JevClient  # noqa: E402
from tabletop.locate import coarse_to_fine  # noqa: E402
from tabletop.scene import (  # noqa: E402
    BLOCK_HALF,
    COLORS,
    DEFAULT_INSTRUCTION,
    TRAY_HALF,
    layout_from_env,
    load_model,
    pixel_to_plane,
    plane_to_pixel,
)

INSTRUCTION = os.environ.get("INSTRUCTION", DEFAULT_INSTRUCTION)
MODE = os.environ.get("PERCEPTION", "jev")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))
# Scripted disturbance: during pick number DISTURB_STEP a "person" slides the target cube.
DISTURB_STEP = int(os.environ.get("DISTURB_STEP", "-1"))
DISTURB_DXY = [float(v) for v in os.environ.get("DISTURB_DXY", "0.0,0.09").split(",")]
HELD_THRESHOLD = 0.5
PLACE_OFFSETS = [(-0.035, 0.035), (0.035, -0.035), (0.035, 0.035), (-0.035, -0.035)]
DRAW_RGB = {"red": (230, 40, 40), "green": (40, 190, 60), "blue": (50, 90, 240),
            "yellow": (230, 190, 0), "tray": (90, 90, 100)}


class Stop(Exception):
    pass


class Brain:
    def __init__(self, node: Node, run_dir: Path):
        self.node = node
        self.run_dir = run_dir
        self.model, self.data = load_model(layout_from_env())
        self.jev = JevClient(MODE, run_dir / "jev_log.jsonl")
        self.metrics = open(run_dir / "metrics.jsonl", "w")  # noqa: SIM115
        self.n_placed = 0

    # ------------------------------------------------------------- dora io
    def wait(self, input_id: str) -> dict:
        while True:
            ev = self.node.next()
            if ev is None or ev["type"] == "STOP":
                raise Stop
            if ev["type"] == "INPUT" and ev["id"] == input_id:
                return ev

    def render(self, camera: str, size: int, tag: str) -> tuple[bytes, dict]:
        self.node.send_output("render_req", pa.array([json.dumps(
            {"camera": camera, "size": size, "tag": tag})]))
        ev = self.wait("camera_image")
        return ev["value"].to_numpy().tobytes(), json.loads(ev["metadata"]["gt"])

    def skill(self, name: str, xy=None) -> None:
        msg: dict[str, Any] = {"name": name}
        if xy is not None:
            msg["xy"] = [float(xy[0]), float(xy[1])]
        self.node.send_output("skill", pa.array([json.dumps(msg)]))
        self.wait("skill_done")

    def overlay(self, title: str, subtitle: str = "", lines=(), image: Image.Image | None = None,
                hold: float = 0.0, panel: str | None = None) -> None:
        ov: dict[str, Any] = {"title": title, "subtitle": subtitle, "lines": list(lines),
                              "hold": hold}
        if panel:
            ov["panel"] = panel
        if image is not None:
            buf = io.BytesIO()
            image.convert("RGB").resize((424, 424)).save(buf, format="PNG")
            ov["image_png"] = base64.b64encode(buf.getvalue()).decode()
        self.node.send_output("overlay", pa.array([json.dumps(ov)]))

    def metric(self, rec: dict) -> None:
        self.metrics.write(json.dumps(rec) + "\n")
        self.metrics.flush()

    # ------------------------------------------------------------ geometry
    def gt_view(self, gt: dict) -> dict:
        """Ground-truth boxes (image fractions) + in-tray flags, for oracle mode & evaluation."""
        boxes = {}
        for c, p in gt["blocks"].items():
            corners = [np.array(p) + np.array(s) * BLOCK_HALF
                       for s in itertools.product([-1, 1], repeat=3)]
            uv = np.array([plane_to_pixel(self.model, self.data, q) for q in corners])
            boxes[c] = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
        tx, ty, _ = gt["tray"]
        hx, hy = TRAY_HALF
        uv = np.array([plane_to_pixel(self.model, self.data, np.array([tx + sx * hx, ty + sy * hy, 0.03]))
                       for sx in (-1, 1) for sy in (-1, 1)])
        boxes["tray"] = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())
        in_tray = {c: bool(abs(p[0] - tx) < hx and abs(p[1] - ty) < hy and p[2] < 0.08)
                   for c, p in gt["blocks"].items()}
        return {"boxes": boxes, "in_tray": in_tray}

    def box_to_world(self, box, z: float) -> np.ndarray:
        u = (box[0] + box[2]) / 2
        v = (box[1] + box[3]) / 2
        return pixel_to_plane(self.model, self.data, u, v, z)

    # ---------------------------------------------------------------- loop
    def run(self, instruction: str = INSTRUCTION) -> None:
        self.overlay("Starting", instruction)
        self.skill("park")
        last_placed: str | None = None
        tray_xy = None
        step = 0
        for step in range(MAX_STEPS):
            png, gt = self.render("overhead", 768, f"overview_{step}")
            gtv = self.gt_view(gt)
            ov = self.jev.overview(png, instruction, last_placed, {**gt, **gtv})
            img = Image.open(io.BytesIO(png)).convert("RGB")
            draw_boxes(img, ov["boxes"])
            probs = sorted(ov["next_probs"].items(), key=lambda kv: -kv[1])
            lines = [f"Jev overview query: {ov['latency_s'] * 1000:.0f} ms ({MODE})",
                     "Choice next: " + "  ".join(f"{k} {p:.2f}" for k, p in probs[:4])]
            lines.append("Noul in tray: " + "  ".join(
                f"{c} {ov['in_tray'][c]:.2f}" for c in COLORS))
            if last_placed:
                ok = ov["in_tray"][last_placed]
                lines.append(f"Check: {last_placed} placed in tray?  p={ok:.2f}")
                self.metric({"event": "place_check", "color": last_placed, "jev": ok,
                             "gt": gtv["in_tray"][last_placed]})
            for c in [*COLORS, "tray"]:
                err = _center_err(ov["boxes"].get(c), gtv["boxes"][c])
                self.metric({"event": "overview_box", "step": step, "target": c,
                             "center_err_frac": err, "conf": ov["box_conf"].get(c)})
            self.metric({"event": "decision", "step": step, "next": ov["next"],
                         "probs": ov["next_probs"], "gt_in_tray": gtv["in_tray"]})
            nxt = ov["next"]
            if tray_xy is None:
                tray_xy = self.box_to_world(ov["boxes"]["tray"], 0.0)[:2]
            if nxt == "done":
                self.overlay("Jev: the instruction is satisfied -> done", instruction, lines, img,
                             hold=3.5)
                break
            self.overlay(f"Jev decides: pick the {nxt} cube next (p={ov['next_probs'][nxt]:.2f})",
                         instruction, lines, img, hold=2.5)

            # ---- coarse-to-fine: re-ask Jev Box on zoomed crops for a precise grasp point
            big, _ = self.render("overhead", 1280, f"refine_{step}")
            bimg = Image.open(io.BytesIO(big)).convert("RGB")
            bx = ov["boxes"][nxt]
            final_box, trace = coarse_to_fine(
                bimg, bx, lambda png, g, c=nxt: self.jev.refine(png, c, g), gt=gtv["boxes"][nxt])
            target = self.box_to_world(final_box, BLOCK_HALF * 1.5)
            true = np.array(gt["blocks"][nxt])
            err_mm = float(np.linalg.norm(target[:2] - true[:2]) * 1000)
            coarse = self.box_to_world(bx, BLOCK_HALF * 1.5)
            coarse_mm = float(np.linalg.norm(coarse[:2] - true[:2]) * 1000)
            self.metric({"event": "grasp_point", "step": step, "color": nxt,
                         "coarse_err_mm": coarse_mm, "refined_err_mm": err_mm,
                         "rounds": [{k: v for k, v in t.items() if k != "crop_img"} for t in trace]})
            for t in trace:
                crop = t["crop_img"]
                draw_boxes(crop, {nxt: t["crop_box"]})
                lat = (t["latency_s"] or 0) * 1000
                verdict = ("accepted" if t["accepted"]
                           else "LOW confidence -> rejected, widen the crop and ask again")
                self.overlay(
                    f"Jev Box, query {t['round'] + 1}/{len(trace)}: locating the {nxt} cube",
                    f"crop = {t['window'][2] * 100:.0f}% of the overhead image; "
                    f"confidence {t['conf'] or 0:.2f} -> {verdict}",
                    [f"Box query: {lat:.0f} ms, confidence {t['conf'] or 0:.2f}",
                     "Confidence gate: accept only if confidence >= 0.30"], crop, hold=1.2,
                    panel="What Jev sees (zoomed crop of the overhead camera)")
            self.overlay(
                f"Grasp point x={target[0]:.3f} m, y={target[1]:.3f} m",
                f"error vs. simulator truth: {err_mm:.0f} mm (before zoom: {coarse_mm:.0f} mm)",
                ["Box centre -> floor plane via the camera model", "Skill: pick(x, y)"],
                hold=1.5)

            # ---- pick, then verify the grasp with the wrist camera
            if step == DISTURB_STEP:
                self.overlay(f"Disturbance: a person slides the {nxt} cube away!",
                             "The robot does not know - it still goes to the old grasp point")
                time.sleep(0.5)  # let the banner reach the recorder before the cube moves
                self.node.send_output("skill", pa.array([json.dumps(
                    {"name": "pick", "xy": [float(target[0]), float(target[1])]})]))
                self.node.send_output("disturb", pa.array([json.dumps(
                    {"color": nxt, "dxy": DISTURB_DXY, "duration": 1.2})]))
                self.metric({"event": "disturbance", "step": step, "color": nxt,
                             "dxy": DISTURB_DXY})
                self.wait("skill_done")
            else:
                self.overlay(f"Executing pick({nxt})", "IK + Cartesian motion (controller node)")
                self.skill("pick", target[:2])
            fpng, fgt = self.render("wrist", 640, f"held_{step}")
            gt_held = fgt["blocks"][nxt][2] > 0.1
            hd = self.jev.held(fpng, nxt, gt_held)
            self.metric({"event": "held_check", "step": step, "color": nxt, "jev": hd["held"],
                         "gt": gt_held})
            held_ok = hd["held"] >= HELD_THRESHOLD
            self.overlay(
                f"Jev Noul: holding the {nxt} cube?  p={hd['held']:.2f} -> "
                + ("yes, go to tray" if held_ok else "no, retry"),
                "two Nouls averaged: 'grasping the cube?' and 'cube between the fingers?'",
                [f"Query: {hd['latency_s'] * 1000:.0f} ms",
                 f"grasping {hd.get('grasping', hd['held']):.2f}   between {hd.get('between', hd['held']):.2f}"], Image.open(io.BytesIO(fpng)),
                hold=2.0, panel="What Jev sees (wrist camera on the hand)")
            if not held_ok:
                self.skill("open")
                self.skill("park")
                last_placed = None
                continue
            off = PLACE_OFFSETS[self.n_placed % len(PLACE_OFFSETS)]
            self.overlay(f"Executing place({nxt}) into the tray", "")
            self.skill("place", (tray_xy[0] + off[0], tray_xy[1] + off[1]))
            self.skill("park")
            self.n_placed += 1
            last_placed = nxt
        self.metric({"event": "end", "steps": step + 1})
        self.overlay("Finished", instruction, hold=1.0)


def _center_err(box, gt_box) -> float | None:
    if box is None:
        return None
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    g = np.array([(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2])
    return float(np.linalg.norm(c - g))


def draw_boxes(img: Image.Image, boxes: dict) -> None:
    d = ImageDraw.Draw(img)
    W, H = img.size
    for name, b in boxes.items():
        if b is None:
            continue
        col = DRAW_RGB.get(name, (255, 255, 255))
        d.rectangle((b[0] * W, b[1] * H, b[2] * W, b[3] * H), outline=col, width=max(2, W // 200))
        d.text((b[0] * W + 2, max(0, b[1] * H - 14)), name, fill=col)


def interactive(brain: Brain) -> None:
    """Wait for instructions typed/spoken in the web UI; run one episode per instruction."""
    brain.skill("park")
    while True:
        brain.overlay("Waiting for your instruction...", "type or speak in the web UI")
        ev = brain.wait("command")
        cmd = json.loads(ev["value"][0].as_py())
        if cmd.get("reset") is not None:
            brain.node.send_output("reset", pa.array([json.dumps({"seed": int(cmd["reset"])})]))
            brain.n_placed = 0
            continue
        text = (cmd.get("text") or "").strip()
        if not text:
            continue
        brain.metric({"event": "instruction", "text": text})
        t0 = time.time()
        brain.run(text)
        brain.metric({"event": "wall_time_s", "value": time.time() - t0, "text": text})


def main() -> None:
    run_dir = Path(os.environ.get("RUN_DIR", "outputs/latest"))
    run_dir.mkdir(parents=True, exist_ok=True)
    node = Node()
    brain = Brain(node, run_dir)
    t0 = time.time()
    try:
        if os.environ.get("INTERACTIVE", "") == "1":
            interactive(brain)
        else:
            brain.run()
            brain.metric({"event": "wall_time_s", "value": time.time() - t0})
    except Stop:
        return
    finally:
        # Tell the simulator to finalise the video and exit; the rest of the graph follows.
        node.send_output("finish", pa.array([True]))


if __name__ == "__main__":
    main()
