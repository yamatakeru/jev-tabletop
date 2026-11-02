# jev-tabletop

Zero-shot tabletop sorting with a simulated Franka Panda, where **all perception and
task-level decisions come from TypeSafe Jev** (a "System One" decision model with image
input). No training, no demonstrations.

```
 overhead camera ──► Jev: Box x4  (an object the instruction wants moved, still outside
                                   the tray; 4 spatial tie-breakers -> up to 4 candidates)
                     Jev: Noul    (anything left to move? / last object in the tray?)
   close-ups     ──► Jev: Noul    (object at the centre? wanted by the instruction?)
   zoomed crops  ──► Jev: Box     (the object that looks like the chosen close-up)
 wrist camera    ──► Jev: Noul    (is the gripper holding an object?)
                         │
                         ▼
          skills (pick / place / park)  ──►  IK + Cartesian motion  ──►  MuJoCo
```

**Open vocabulary**: the code never names an object or colour in a Jev question. The
instruction is passed as text and every question refers to "an object that `instruction`
asks to put into the tray"; once chosen, the object is tracked by appearance (a close-up
passed as a `reference` image). The closed-vocabulary version that listed four colours is
tagged `closed-vocab-v3`. Object names in logs/videos come from the simulator
(display/evaluation only).

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

### Interactive demo (talk to the robot)

```bash
INTERACTIVE=1 LIVE=1 RUN_DIR=outputs/interactive uv run dora run dataflow_interactive.yml
# open http://127.0.0.1:8765 : live view, type an instruction (or 🎤 speak, EN/JA), "Shuffle cubes"
```

### Recovery from a disturbance

```bash
# During the 2nd pick a "person" slides the target object 9 cm; the wrist-camera Noul
# detects the failed grasp and Jev re-localises it.
RUN_DIR=outputs/demoC DISTURB_STEP=1 uv run dora run dataflow.yml
```

### Other objects

```bash
# OBJECTS=colour[:shape],...  colours: red green blue yellow purple orange; shapes: cube, cylinder
OBJECTS=red,green,blue,yellow,purple,orange:cylinder LAYOUT_SEED=12 \
INSTRUCTION="Tidy away the round object." RUN_DIR=outputs/round uv run dora run dataflow.yml
```

### Experiments

```bash
# development set (5 layouts + 6 instructions); STOP_CONF=none logs two zoom rounds per pick
EXP_NAME=dev_ov STOP_CONF=none uv run python scripts/experiments.py dev
EXP_NAME=dev_ov uv run python scripts/experiments.py summary
uv run python scripts/zoom_analysis.py outputs/dev_ov   # adaptive-zoom threshold (dev data only)
uv run python scripts/heldout.py run new|old       # pre-registered held-out conditions
uv run python scripts/heldout.py summary           # -> outputs/heldout_summary.md
```

### Reproducing the report

Jev model: `jev-1.14.0` (default of `JEV_MODEL`; was `jev-latest` until after the freeze,
which resolved to `jev-1.14.0` in every logged run). Tags mark the code of each stage; run
older stages in a separate worktree (`git worktree add ../jt-v1 exp-v1`, then symlink
`assets/menagerie` and `.env`).

| report table | tag | command |
|---|---|---|
| v1 / v2 / v3 comparison (5 layouts, 6 instructions) | `exp-v1`, `exp-v2` (IK fix, no gate), `exp-v3` (confidence gate) | `uv run python scripts/experiments.py layouts && uv run python scripts/experiments.py instructions && uv run python scripts/experiments.py summary` |
| traffic-light instruction (English, seed 3) | `closed-vocab-v3` | `LAYOUT_SEED=3 INSTRUCTION="Put every cube whose colour appears on a traffic light into the tray." RUN_DIR=outputs/demoB uv run dora run dataflow.yml` |
| open-vocabulary version on the development set (seeds 1,2,4,5,6 + 7 instructions) | `main` | `EXP_NAME=exp_ov_final uv run python scripts/experiments.py final && EXP_NAME=exp_ov_final uv run python scripts/experiments.py summary` |
| adaptive-zoom threshold | `openvocab-frozen` | `EXP_NAME=dev_ov STOP_CONF=none uv run python scripts/experiments.py dev`, then `uv run python scripts/zoom_analysis.py outputs/dev_ov` |
| held-out, new vs. old | `openvocab-frozen` + `closed-vocab-v3` (in `../jev-tabletop-closed`, with the frozen `tabletop/scene.py` copied in) | `uv run python scripts/heldout.py run new && uv run python scripts/heldout.py run old && uv run python scripts/heldout.py summary` |

Layout seeds: 1-6 development (3 = demo B), 7-17 held-out. Jev answers are not bit-exact
across runs (the rendered scene differs slightly as the robot moves), so repeated runs can
differ in individual decisions.

Each run writes `run.mp4` (demo video), `jev_log.jsonl` (every Jev request/answer and
latency) and `metrics.jsonl` (decisions, grasp-point error vs. ground truth, checks).

## Layout

```
dataflow.yml          dora graph
nodes/sim.py          MuJoCo physics, overhead/front/wrist cameras, video recorder
nodes/controller.py   skills -> Cartesian waypoints -> IK -> joint targets
nodes/brain.py        task loop: Jev decisions -> skills, verification, metrics
nodes/ui.py           web UI for the interactive demo (MJPEG stream + instruction box)
tabletop/scene.py     scene, cameras, pixel <-> floor-plane geometry
tabletop/jev.py       Jev questions (Box / Noul, open vocabulary) + oracle baseline
tabletop/locate.py    coarse-to-fine Box localisation on zoomed crops (confidence gate,
                      adaptive depth)
tabletop/ik.py        damped-least-squares IK
tabletop/video.py     video composition (robot view + what Jev saw + banner)
```
