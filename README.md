# jev-tabletop

Zero-shot tabletop sorting with a simulated Franka Panda, where **all perception and
task-level decisions come from TypeSafe Jev** (a "System One" decision model with image
input). No training, no demonstrations.

```
 overhead camera ──► Jev: Choice  (which cube next / done?)
                     Jev: Box     (where is it? coarse -> zoomed crops)
                     Jev: Noul    (is it already in the tray?)
 wrist camera    ──► Jev: Noul    (is the gripper holding it?)
                         │
                         ▼
          skills (pick / place / park)  ──►  IK + Cartesian motion  ──►  MuJoCo
```

The nodes are wired as a [dora-rs](https://github.com/dora-rs/dora) dataflow
(`dataflow.yml`): `sim` (MuJoCo + cameras + video), `controller` (skills, IK),
`brain` (Jev queries, task loop).

Only sensors that a real cell would have are used for control: a fixed overhead camera,
a wrist camera rigidly mounted on the hand, and joint encoders. Simulator ground truth is
used only to log evaluation metrics (and in the `oracle` baseline mode).

## Setup

```bash
uv sync
./scripts/fetch_assets.sh          # Franka Panda from MuJoCo Menagerie (Apache-2.0)
cp .env.example .env               # put your TYPESAFE_API_KEY in .env
```

## Run

```bash
# Explicit instruction (default layout)
RUN_DIR=outputs/demoA uv run dora run dataflow.yml

# Semantic instruction + random layout
RUN_DIR=outputs/demoB LAYOUT_SEED=3 \
INSTRUCTION="Put every cube whose colour appears on a traffic light into the tray." \
  uv run dora run dataflow.yml

# Baseline with simulator ground truth instead of Jev
PERCEPTION=oracle RUN_DIR=outputs/oracle uv run dora run dataflow.yml
```

Each run writes `run.mp4` (demo video), `jev_log.jsonl` (every Jev request/answer and
latency) and `metrics.jsonl` (decisions, grasp-point error vs. ground truth, checks).

## Layout

```
dataflow.yml          dora graph
nodes/sim.py          MuJoCo physics, overhead/front/wrist cameras, video recorder
nodes/controller.py   skills -> Cartesian waypoints -> IK -> joint targets
nodes/brain.py        task loop: Jev decisions -> skills, verification, metrics
tabletop/scene.py     scene, cameras, pixel <-> floor-plane geometry
tabletop/jev.py       Jev questions (Choice / Box / Noul) + oracle baseline
tabletop/locate.py    coarse-to-fine Box localisation on zoomed crops
tabletop/ik.py        damped-least-squares IK
tabletop/video.py     video composition (robot view + what Jev saw + banner)
```
