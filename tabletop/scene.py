# pyright: reportAttributeAccessIssue=false
"""Tabletop scene: Franka Panda (MuJoCo Menagerie) + colored blocks + a tray.

The scene XML is generated next to the Menagerie ``panda.xml`` so that its relative
mesh paths resolve. Everything a node needs (model loading, camera geometry, the
tool-centre-point) lives here so that sim, controller and brain agree on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_DIR = ROOT / "assets" / "menagerie" / "franka_emika_panda"

BLOCK_HALF = 0.022  # 4.4 cm cubes
TRAY_HALF = (0.09, 0.09)  # inner half size (x, y)
TRAY_WALL_H = 0.035

DEFAULT_INSTRUCTION = (
    "Put the red, blue and green cubes into the tray. Leave the yellow cube alone."
)

# Distinct, easy-to-name colours. Names are what the instruction / Jev talk about.
COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.12, 0.12),
    "green": (0.15, 0.7, 0.2),
    "blue": (0.15, 0.3, 0.9),
    "yellow": (0.95, 0.8, 0.1),
}

# Overhead camera used for perception (looks straight down).
OVERHEAD_POS = (0.5, 0.0, 1.25)
OVERHEAD_FOVY = 50.0
# Video camera (perspective, for the human viewer only).
FRONT_POS = (1.45, 0.5, 0.95)
FRONT_LOOKAT = (0.3, -0.08, 0.22)
# Arm "parked" out of the overhead camera's view while perceiving.
PARK_Q = (1.9, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785)
# Gripper pointing down, fingers closing along world y (matches the home pose).
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# Lowest point the tool centre point may reach when grasping (just above the floor).
TCP_OFFSET = 0.1034  # hand frame origin -> fingertip centre along hand z


@dataclass(frozen=True)
class Layout:
    blocks: dict[str, tuple[float, float]]
    tray: tuple[float, float]


DEFAULT_LAYOUT = Layout(
    blocks={
        "red": (0.62, 0.12),
        "green": (0.48, -0.05),
        "blue": (0.66, -0.18),
        "yellow": (0.44, 0.2),
    },
    tray=(0.40, -0.32),
)


def layout_from_env() -> Layout:
    seed = os.environ.get("LAYOUT_SEED", "")
    return random_layout(int(seed)) if seed else DEFAULT_LAYOUT


def random_layout(seed: int) -> Layout:
    """Random non-overlapping block positions in front of the robot."""
    rng = np.random.default_rng(seed)
    tray = (0.40, -0.32)
    placed: list[np.ndarray] = []
    blocks: dict[str, tuple[float, float]] = {}
    for name in COLORS:
        p = np.array([0.5, 0.0])
        for _ in range(1000):
            p = np.array([rng.uniform(0.40, 0.68), rng.uniform(-0.18, 0.28)])
            if all(np.linalg.norm(p - q) > 0.09 for q in placed):
                break
        placed.append(p)
        blocks[name] = (float(p[0]), float(p[1]))
    return Layout(blocks=blocks, tray=tray)


def _tray_xml(x: float, y: float) -> str:
    hx, hy = TRAY_HALF
    t = 0.006  # wall thickness
    h = TRAY_WALL_H / 2
    walls = [
        (0, hy + t, hx + t, t),
        (0, -hy - t, hx + t, t),
        (hx + t, 0, t, hy + t),
        (-hx - t, 0, t, hy + t),
    ]
    geoms = [
        f'<geom type="box" size="{hx + t} {hy + t} 0.003" pos="0 0 0.003" rgba="0.92 0.92 0.92 1"/>'
    ]
    for wx, wy, sx, sy in walls:
        geoms.append(
            f'<geom type="box" size="{sx} {sy} {h}" pos="{wx} {wy} {h}" rgba="0.55 0.55 0.6 1"/>'
        )
    return f'<body name="tray" pos="{x} {y} 0">{"".join(geoms)}</body>'


def _look_at_xyaxes(pos, target) -> np.ndarray:
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.concatenate([right, up])


def build_scene_xml(layout: Layout) -> str:
    blocks = []
    for name, (x, y) in layout.blocks.items():
        r, g, b = COLORS[name]
        blocks.append(
            f'<body name="block_{name}" pos="{x} {y} {BLOCK_HALF}">'
            f'<freejoint name="block_{name}_joint"/>'
            f'<geom type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" mass="0.05" '
            f'rgba="{r} {g} {b} 1" friction="1.5 0.01 0.001" condim="4" solref="0.01 1"/>'
            f"</body>"
        )
    cx, cy, cz = OVERHEAD_POS
    fx, fy, fz = FRONT_POS
    front_xyaxes = " ".join(f"{v:.4f}" for v in _look_at_xyaxes(FRONT_POS, FRONT_LOOKAT))
    return f"""<mujoco model="jev tabletop">
  <include file="panda.xml"/>
  <option timestep="0.002" impratio="10" cone="elliptic"/>
  <visual>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-30" offwidth="1280" offheight="1280"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.75 0.82 0.9" rgb2="0.3 0.35 0.45" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.78 0.72 0.62" rgb2="0.72 0.66 0.56" width="300" height="300"/>
    <material name="floor" texture="floor" texrepeat="8 8" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0.5 0 2" dir="0.3 0.2 -1" directional="true" diffuse="0.45 0.45 0.45" castshadow="true"/>
    <light pos="1.2 1.0 1.6" dir="-0.5 -0.5 -1" diffuse="0.15 0.15 0.15" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor"/>
    <camera name="overhead" pos="{cx} {cy} {cz}" xyaxes="0 -1 0 1 0 0" fovy="{OVERHEAD_FOVY}"/>
    <camera name="front" pos="{fx} {fy} {fz}" xyaxes="{front_xyaxes}" fovy="45"/>
    {_tray_xml(*layout.tray)}
    {"".join(blocks)}
  </worldbody>
</mujoco>
"""


# Wrist camera rigidly mounted on the hand (like a RealSense on a real Panda flange),
# offset from the gripper axis and tilted to look at the space between the fingertips.
WRIST_CAM_POS = np.array([0.085, 0.0, 0.0])  # hand frame
WRIST_CAM_LOOK = np.array([0.0, 0.0, 0.115])  # hand frame point between the fingertips
WRIST_CAM_FOVY = 65.0


def _add_wrist_camera(spec) -> None:
    look = WRIST_CAM_LOOK - WRIST_CAM_POS
    z_cam = -look / np.linalg.norm(look)  # MuJoCo cameras look along -z
    # Image "up" = away from the hand body, so the hand sits at the bottom of the image.
    up = np.array([1.0, 0.0, 0.0])
    y_cam = up - up.dot(z_cam) * z_cam
    y_cam /= np.linalg.norm(y_cam)
    x_cam = np.cross(y_cam, z_cam)
    R = np.column_stack([x_cam, y_cam, z_cam])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    spec.body("hand").add_camera(name="wrist", pos=WRIST_CAM_POS, quat=quat,
                                 fovy=WRIST_CAM_FOVY)


def load_model(layout: Layout = DEFAULT_LAYOUT) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not (PANDA_DIR / "panda.xml").exists():
        raise FileNotFoundError("Run scripts/fetch_assets.sh first (Menagerie Panda missing).")
    # Per-process file: several dora nodes load the model concurrently.
    path = PANDA_DIR / f"tabletop_scene_{os.getpid()}.xml"
    path.write_text(build_scene_xml(layout))
    try:
        spec = mujoco.MjSpec.from_file(str(path))
    finally:
        path.unlink(missing_ok=True)
    _add_wrist_camera(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    key = model.key("home").id
    # The keyframe only covers the robot; keep blocks at their spawn pose.
    nq_robot = 9
    data.qpos[:nq_robot] = model.key_qpos[key][:nq_robot]
    data.qpos[:7] = PARK_Q
    data.ctrl[:] = model.key_ctrl[key]
    data.ctrl[:7] = PARK_Q
    mujoco.mj_forward(model, data)
    return model, data


# ---------------------------------------------------------------- geometry helpers


def tcp_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Tool centre point position and rotation matrix (world)."""
    hand = model.body("hand").id
    R = data.xmat[hand].reshape(3, 3)
    return data.xpos[hand] + R @ np.array([0, 0, TCP_OFFSET]), R.copy()


def block_positions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, list[float]]:
    return {n: data.xpos[model.body(f"block_{n}").id].round(4).tolist() for n in COLORS}


def tray_position(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    return data.xpos[model.body("tray").id].round(4).tolist()


def pixel_to_plane(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    u: float,
    v: float,
    z_plane: float,
    camera: str = "overhead",
) -> np.ndarray:
    """Intersect the camera ray through image fraction (u right, v down) with z = z_plane."""
    cam = model.camera(camera).id
    pos = data.cam_xpos[cam]
    R = data.cam_xmat[cam].reshape(3, 3)  # columns: camera x (right), y (up), z (backward)
    fovy = np.deg2rad(model.cam_fovy[cam])
    t = np.tan(fovy / 2)
    # Square images assumed for perception (aspect 1).
    x_cam = (2 * u - 1) * t
    y_cam = (1 - 2 * v) * t
    ray = R @ np.array([x_cam, y_cam, -1.0])
    s = (z_plane - pos[2]) / ray[2]
    return pos + s * ray


def plane_to_pixel(model, data, p: np.ndarray, camera: str = "overhead") -> tuple[float, float]:
    cam = model.camera(camera).id
    pos = data.cam_xpos[cam]
    R = data.cam_xmat[cam].reshape(3, 3)
    t = np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
    c = R.T @ (np.asarray(p) - pos)
    x, y = c[0] / -c[2], c[1] / -c[2]
    return (x / t + 1) / 2, (1 - y / t) / 2
