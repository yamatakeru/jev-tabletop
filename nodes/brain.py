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
from tabletop.jev import VERIFY_CROP, JevClient
from tabletop.locate import coarse_to_fine, crop_png, crop_window
from tabletop.scene import (
    BLOCK_HALF,
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
# Adaptive zoom depth: stop after one accepted zoom if its Box confidence >= STOP_CONF;
# otherwise zoom once more (and low-confidence answers are rejected by the gate).
# Chosen on development runs only (scripts/zoom_analysis.py): 42 picks, all first answers
# >= 0.52; one zoom vs two: median 6.7 vs 6.5 mm, max 16.6 vs 17.0 mm, queries 1.0 vs 2.0.
# STOP_CONF=none -> always two accepted rounds (how the development data was collected).
_STOP = os.environ.get("STOP_CONF", "0.5")
STOP_CONF = None if _STOP.lower() == "none" else float(_STOP)


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
        self.img_dir = run_dir / "images"  # what Jev saw at the held check (for inspection)
        self.img_dir.mkdir(exist_ok=True)

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

    def sim_label(self, box, gtv: dict) -> str:
        """Simulator object nearest to an image box: evaluation / display only."""
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        names = [n for n in gtv["boxes"] if n != "tray"]
        return min(names, key=lambda n: np.linalg.norm(c - _centre(gtv["boxes"][n])))

    # ---------------------------------------------------------------- loop
    def run(self, instruction: str = INSTRUCTION) -> None:
        self.overlay("Starting", instruction)
        self.skill("park")
        last_label: str | None = None  # simulator label of the last placed object (eval only)
        last_ref: bytes | None = None  # Jev-side identity of the last placed object: a close-up
        tray_xy = None
        step = 0
        for step in range(MAX_STEPS):
            png, gt = self.render("overhead", 768, f"overview_{step}")
            gtv = self.gt_view(gt)
            ov = self.jev.overview(png, instruction, last_ref, {**gt, **gtv})
            img = Image.open(io.BytesIO(png)).convert("RGB")
            for cd in ov["candidates"]:
                cd["sim_label"] = self.sim_label(cd["box"], gtv)
            label = self.sim_label(ov["target"], gtv) if ov["target"] else None
            draw_boxes(img, {"tray": ov["tray"]})
            for cd in ov["candidates"]:
                draw_boxes(img, {cd["var"]: cd["box"]}, (160, 160, 160))
            if ov["target"]:
                draw_boxes(img, {"target": ov["target"]}, (255, 0, 200))
            lines = [f"Jev overview: {ov['latency_s'] * 1000:.0f} ms, {ov['n_queries']} queries ({MODE})",
                     f"Noul 'anything left to move?' p={ov['remaining']:.2f}"]
            if ov["candidates"] and "score" in ov["candidates"][0]:
                lines.append("Candidates (Box) verified by Noul on close-ups: " + "  ".join(
                    f"{cd['score']:.2f}" for cd in ov["candidates"]))
            if last_label is not None and ov["placed_ok"] is not None:
                lines.append(f"Check: last object in the tray?  p={ov['placed_ok']:.2f}")
                self.metric({"event": "place_check", "color": last_label, "jev": ov["placed_ok"],
                             "gt": gtv["in_tray"][last_label]})
            self.metric({"event": "overview_box", "step": step, "target": "tray",
                         "center_err_frac": _center_err(ov["tray"], gtv["boxes"]["tray"])})
            self.metric({"event": "decision", "step": step, "next": label or "done",
                         "remaining": ov["remaining"], "n_queries": ov["n_queries"],
                         "latency_s": ov["latency_s"],
                         "candidates": [{k: v for k, v in cd.items() if k != "png"}
                                        for cd in ov["candidates"]],
                         "gt_in_tray": gtv["in_tray"]})
            if tray_xy is None:
                tray_xy = self.box_to_world(ov["tray"], 0.0)[:2]
            if label is None:
                self.overlay("Jev: the instruction is satisfied -> done", instruction, lines, img,
                             hold=3.5)
                break
            self.overlay(f"Jev picks the next object ({label} - label from simulator, display only)",
                         instruction, lines, img, hold=2.5)

            # ---- coarse-to-fine: re-ask Jev Box on zoomed crops, tracking the chosen object by
            # its appearance (close-up reference image), not by a name
            big, _ = self.render("overhead", 1280, f"refine_{step}")
            bimg = Image.open(io.BytesIO(big)).convert("RGB")
            ref = ov["ref_png"]
            final_box, trace = coarse_to_fine(
                bimg, ov["target"], lambda png, g, r=ref: self.jev.refine(png, r, g),
                gt=gtv["boxes"][label], stop_conf=STOP_CONF)
            # Close-up of the localised object: its identity for the held / placed checks.
            cu, cv = _centre(final_box)
            l0, t0 = crop_window((cu, cv, cu, cv), VERIFY_CROP)
            ref, _ = crop_png(bimg, l0, t0, VERIFY_CROP)
            target = self.box_to_world(final_box, BLOCK_HALF * 1.5)
            true = np.array(gt["blocks"][label])
            err_mm = float(np.linalg.norm(target[:2] - true[:2]) * 1000)
            coarse = self.box_to_world(ov["target"], BLOCK_HALF * 1.5)
            coarse_mm = float(np.linalg.norm(coarse[:2] - true[:2]) * 1000)
            rounds = []
            for t in trace:
                rec = {k: v for k, v in t.items() if k != "crop_img"}
                w = self.box_to_world(t["full_box"], BLOCK_HALF * 1.5)
                rec["err_mm"] = float(np.linalg.norm(w[:2] - true[:2]) * 1000)
                rounds.append(rec)
            self.metric({"event": "grasp_point", "step": step, "color": label,
                         "coarse_err_mm": coarse_mm, "refined_err_mm": err_mm, "rounds": rounds})
            for t in trace:
                crop = t["crop_img"]
                draw_boxes(crop, {"target": t["crop_box"]}, (255, 0, 200))
                lat = (t["latency_s"] or 0) * 1000
                verdict = ("accepted" if t["accepted"]
                           else "LOW confidence -> rejected, widen the crop and ask again")
                self.overlay(
                    f"Jev Box, query {t['round'] + 1}/{len(trace)}: the object that looks like "
                    "the chosen one",
                    f"crop = {t['window'][2] * 100:.0f}% of the overhead image; "
                    f"confidence {t['conf'] or 0:.2f} -> {verdict}",
                    [f"Box query: {lat:.0f} ms, confidence {t['conf'] or 0:.2f}",
                     "Confidence gate: accept only if confidence >= 0.30",
                     f"Adaptive zoom: stop once an accepted answer has confidence >= {STOP_CONF}"],
                    crop, hold=1.2,
                    panel="What Jev sees (zoomed crop of the overhead camera)")
            self.overlay(
                f"Grasp point x={target[0]:.3f} m, y={target[1]:.3f} m",
                f"error vs. simulator truth: {err_mm:.0f} mm (before zoom: {coarse_mm:.0f} mm)",
                ["Box centre -> floor plane via the camera model", "Skill: pick(x, y)"],
                hold=1.5)

            # ---- pick, then verify the grasp with the wrist camera
            if step == DISTURB_STEP:
                self.overlay("Disturbance: a person slides the object away!",
                             "The robot does not know - it still goes to the old grasp point")
                time.sleep(0.5)  # let the banner reach the recorder before the object moves
                self.node.send_output("skill", pa.array([json.dumps(
                    {"name": "pick", "xy": [float(target[0]), float(target[1])]})]))
                self.node.send_output("disturb", pa.array([json.dumps(
                    {"color": label, "dxy": DISTURB_DXY, "duration": 1.2})]))
                self.metric({"event": "disturbance", "step": step, "color": label,
                             "dxy": DISTURB_DXY})
                self.wait("skill_done")
            else:
                self.overlay("Executing pick(x, y)", "IK + Cartesian motion (controller node)")
                self.skill("pick", target[:2])
            fpng, fgt = self.render("wrist", 640, f"held_{step}")
            gt_held = fgt["blocks"][label][2] > 0.1
            (self.img_dir / f"held_{step}.png").write_bytes(fpng)
            hd = self.jev.held(fpng, gt_held)
            self.metric({"event": "held_check", "step": step, "color": label, "jev": hd["held"],
                         "gt": gt_held})
            held_ok = hd["held"] >= HELD_THRESHOLD
            self.overlay(
                f"Jev Noul: holding an object?  p={hd['held']:.2f} -> "
                + ("yes, go to tray" if held_ok else "no, retry"),
                "two Nouls averaged: 'grasping an object?' and 'object between the fingers?'",
                [f"Query: {hd['latency_s'] * 1000:.0f} ms",
                 f"grasping {hd.get('grasping', hd['held']):.2f}   between {hd.get('between', hd['held']):.2f}"], Image.open(io.BytesIO(fpng)),
                hold=2.0, panel="What Jev sees (wrist camera on the hand)")
            if not held_ok:
                self.skill("open")
                self.skill("park")
                last_label, last_ref = None, None
                continue
            off = PLACE_OFFSETS[self.n_placed % len(PLACE_OFFSETS)]
            self.overlay("Executing place() into the tray", "")
            self.skill("place", (tray_xy[0] + off[0], tray_xy[1] + off[1]))
            self.skill("park")
            self.n_placed += 1
            last_label, last_ref = label, ref
        self.metric({"event": "end", "steps": step + 1})
        self.overlay("Finished", instruction, hold=1.0)


def _centre(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def _center_err(box, gt_box) -> float | None:
    if box is None:
        return None
    return float(np.linalg.norm(_centre(box) - _centre(gt_box)))


def draw_boxes(img: Image.Image, boxes: dict, col: tuple = (90, 90, 100)) -> None:
    d = ImageDraw.Draw(img)
    W, H = img.size
    for name, b in boxes.items():
        if b is None:
            continue
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
