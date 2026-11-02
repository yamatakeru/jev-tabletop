# pyright: reportAttributeAccessIssue=false
"""dora node: MuJoCo physics + cameras + video recorder.

inputs
  tick        timer; every tick advances SIM_DT seconds of physics
  joint_cmd   float64[8]: 7 arm joint targets + gripper command (0 closed .. 255 open)
  render_req  JSON string {"camera": "overhead"|"front", "size": int, "tag": str}
  overlay     JSON string describing what the brain decided (drawn into the video)
outputs
  robot_state float64[13]: sim_time, q[7], finger[2], tcp xyz
  camera_image uint8 PNG bytes, metadata: camera, size, tag, gt (JSON ground truth)
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
from dora import Node
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabletop.scene import (  # noqa: E402
    BLOCK_HALF,
    random_layout,
    block_positions,
    layout_from_env,
    load_model,
    tcp_pose,
    tray_position,
)
from tabletop.video import VideoComposer  # noqa: E402

SIM_DT = 0.02  # physics seconds per tick
FRAME_EVERY = 2  # record every 2nd tick -> 25 fps of sim time


def apply_disturbance(model, data, d: dict) -> dict | None:
    """Slide a block along the floor (kinematically) as if a person pushed it."""
    s = min((data.time - d["t0"]) / d["dur"], 1.0)
    s = s * s * (3 - 2 * s)
    data.qpos[d["adr"]:d["adr"] + 3] = d["start"] + (d["end"] - d["start"]) * s
    data.qvel[d["dof"]:d["dof"] + 6] = 0.0
    return None if s >= 1.0 else d


def main() -> None:
    out_dir = Path(os.environ.get("RUN_DIR", "outputs/latest"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model, data = load_model(layout_from_env())
    n_sub = round(SIM_DT / model.opt.timestep)
    renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def render(camera: str, w: int, h: int) -> np.ndarray:
        key = (w, h)
        if key not in renderers:
            renderers[key] = mujoco.Renderer(model, h, w)
        r = renderers[key]
        r.update_scene(data, camera)
        return r.render()

    video = VideoComposer(out_dir / "run.mp4", fps=25)
    if os.environ.get("NO_VIDEO", "") != "1":
        video.title_card(
            os.environ.get("TITLE", "Jev x MuJoCo: zero-shot tabletop sorting"),
            [
                "No training, no demonstrations",
                f"Instruction: {os.environ.get('INSTRUCTION', '')}",
                "Perception & decisions: TypeSafe Jev (System One model, image input)",
            ],
            seconds=4.0,
        )
    node = Node()
    tick = 0
    last_cmd_time = -1.0
    record = os.environ.get("NO_VIDEO", "") != "1"
    live = os.environ.get("LIVE", "") == "1"  # stream composite frames to the web UI
    disturb: dict | None = None  # active scripted disturbance ("a person slides a cube")
    log = open(out_dir / "sim_events.jsonl", "w")  # noqa: SIM115

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        eid = event["id"]
        if eid == "tick":
            if disturb is not None:
                disturb = apply_disturbance(model, data, disturb)
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            tick += 1
            tcp, _ = tcp_pose(model, data)
            state = np.concatenate([[data.time], data.qpos[:9], tcp])
            node.send_output("robot_state", pa.array(state))
            active = data.time - last_cmd_time < 0.1
            if live and tick % 3 == 0:
                buf = io.BytesIO()
                Image.fromarray(video.compose(render("front", 832, 624), data.time)).save(
                    buf, format="JPEG", quality=80)
                node.send_output("live_frame", pa.array(np.frombuffer(buf.getvalue(), dtype=np.uint8)))
            if record and (active or disturb is not None) and tick % FRAME_EVERY == 0:
                video.frame(render("front", 832, 624), sim_time=data.time)
        elif eid == "joint_cmd":
            cmd = event["value"].to_numpy()
            data.ctrl[:8] = cmd[:8]
            last_cmd_time = data.time
        elif eid == "render_req":
            req = json.loads(event["value"][0].as_py())
            size = int(req.get("size", 768))
            img = render(req.get("camera", "overhead"), size, size)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            gt = {
                "blocks": block_positions(model, data),
                "tray": tray_position(model, data),
                "sim_time": round(data.time, 3),
            }
            node.send_output(
                "camera_image",
                pa.array(np.frombuffer(buf.getvalue(), dtype=np.uint8)),
                {"camera": req.get("camera", "overhead"), "size": size,
                 "tag": req.get("tag", ""), "gt": json.dumps(gt)},
            )
            log.write(json.dumps({"wall": time.time(), "render": req, "gt": gt}) + "\n")
        elif eid == "disturb":
            req = json.loads(event["value"][0].as_py())
            jid = model.joint(f"block_{req['color']}_joint").id
            adr = model.jnt_qposadr[jid]
            start = data.qpos[adr:adr + 3].copy()
            disturb = {"adr": adr, "dof": model.jnt_dofadr[jid], "start": start,
                       "end": start + np.array([*req["dxy"], 0.0]), "t0": data.time,
                       "dur": float(req.get("duration", 0.8))}
            log.write(json.dumps({"wall": time.time(), "disturb": req}) + "\n")
        elif eid == "reset":
            seed = json.loads(event["value"][0].as_py())["seed"]
            lay = random_layout(seed)
            for c, (x, y) in lay.blocks.items():
                jid = model.joint(f"block_{c}_joint").id
                adr, dof = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
                data.qpos[adr:adr + 7] = [x, y, BLOCK_HALF, 1, 0, 0, 0]
                data.qvel[dof:dof + 6] = 0.0
            mujoco.mj_forward(model, data)
        elif eid == "finish":
            break
        elif eid == "overlay":
            ov = json.loads(event["value"][0].as_py())
            video.set_overlay(ov)
            if record and ov.get("hold", 0) > 0:
                video.hold(render("front", 832, 624), seconds=float(ov["hold"]), sim_time=data.time)
            log.write(json.dumps({"wall": time.time(), "overlay": {k: v for k, v in ov.items() if k != "image_png"}}) + "\n")
            log.flush()

    final = {"blocks": block_positions(model, data), "tray": tray_position(model, data)}
    (out_dir / "final_state.json").write_text(json.dumps(final, indent=2))
    video.close()
    log.close()


if __name__ == "__main__":
    main()
