# Biddy: autonomous brick inspection cell

Intel Physical AI Challenge, Junction 2026. An SO-101 arm picks each Duplo
brick, shows it to the inspection camera and drops it into the good or
defective bowl. See [AGENTS.md](AGENTS.md) for the task and hardware.

## Quick start

Everything runs through `./biddy` (`./biddy help`), which picks the right venv:
the Studio venv (physicalai, LeRobot, OpenVINO) for the cell, or
`anomaly/.venv` (Anomalib, PyTorch XPU build) for anomaly training.

```bash
./biddy setup        # once: anomaly venv, then shows devices and what Studio has
```

Robot calibration, serial numbers and camera identities come from Studio's
database (`station.yaml` only names them), so the cell drives exactly what the
demos were recorded with. **Close Studio's robot session before anything that
touches the arm or cameras**: Studio locks them while a session is open.

## Intel stack

| Stage | Tool | Device |
|---|---|---|
| Record demos | Physical AI Studio | laptop |
| Train policy | Physical AI Studio (`./biddy train-policy`) | remote trainer (cloud GPU) or local Intel GPU |
| Policy inference | OpenVINO via physicalai runtime | GPU (`policy.device`) |
| Train anomaly model | Anomalib | CPU (XPU available, see note) |
| Optimize anomaly model | OpenVINO export + NNCF INT8 | |
| Anomaly inference | OpenVINO | NPU (`inspection.device`) |
| Control loop | physicalai RobotRuntime | CPU |

The NPU driver here is Canonical's `intel-npu-driver` snap; `cell/intel.py`
points OpenVINO at it automatically (`./biddy devices` shows CPU GPU NPU).
`./biddy bench` times every model on every device and saves the table.
Anomalib training on the XPU currently hits an `xe` driver page fault on this
laptop (`anomaly/train.py --accelerator xpu`); Patchcore on the CPU takes seconds.

## From recorded demos to a running cell

1. **Record** in Studio following [docs/data_collection.md](docs/data_collection.md),
   with exactly two task phrases: `put the brick in the good bowl` and
   `put the brick in the defective bowl` (they must match `policy.tasks`).
   Mix colours, and good and defective bricks. **Finish recording** so Studio
   writes the dataset.
2. `./biddy check`: per episode, holds per inspection spot and teleop jumps;
   per colour, good/defective counts. Delete flagged episodes in Studio.
3. If the inspection spots or arm poses changed, run
   `$PY scripts/find_spots.py` and copy the spot poses into
   `inspection.views.<view>.poses`; set crop boxes with `./biddy roi --view <view> <image>`.
4. `./biddy train-policy --policy smolvla --name smolvla-v1 [--trainer <name>]`.
5. `./biddy poses record --only home` (plus the bowls for scripted drops).
6. `./biddy anomaly`: crops per view from the demos -> train -> INT8 -> benchmark.
7. `./biddy mock --cycles 6` (mechanics only), then `./biddy run --cycles 5 --show`.

Each run writes `runs/<time>/cycles.csv` (verdict, worst face and view, score
vs threshold, holds, timings per brick) and each brick's worst crop.

## Inspection views

`station.yaml` `inspection.views` lists one Anomalib model per view: a crop
box on a camera. The station holds the brick at two spots in the inspection
image (`side`: centre, `bottom`: bottom right) and the wrist camera sees the
brick before the grasp (`wrist`, the hover view). A hold belongs to a view
when the arm is at one of that view's `poses` (from `scripts/find_spots.py`)
and its crop differs from the empty view. Colours and orientations: training
adds rotated/flipped copies of good crops (`anomaly/train.py --augment`), and
thresholds are picked per episode from its worst face, since a defective brick
has its mark on one face only.

## How a cycle works

`cell/run.py`: the policy plays a whole demo per brick. A runtime callback spots
holds and a background thread scores each with its view's model on the NPU.
As soon as one face fails, the policy's task switches to the defective bowl,
which is before the drop. The brick is done when the arm is back home. A run
with no inspection hold was a missed grasp: the policy tries again. With
`policy.drop: scripted` (or a separate `policy.drop_model`), the policy stops
after `inspection.holds_per_brick` holds and the arm replays the bowl pose (or
the drop policy takes over). Every policy action is checked before it reaches
the arm: NaN ends the run, everything else is clamped to the joint range.

## AprilTag check

With the arm at home before every brick, the cell compares the AprilTags in
the top and inspection images with `calibration/reference_<camera>.json`
(`station.yaml` `tags`, `cell/tags.py`; about 0.25 s per brick).

- A bowl, the arm base, the stand or the top camera moved: the cell prints
  what moved and waits until it is back (`on_moved: pause`), or only logs it
  (`warn`). Time spent waiting is not counted as cycle time.
- The inspection camera moved: its crop boxes move with the tags, so the
  Anomalib models still see the brick where they were trained to. The frame
  is not warped: the crop boxes reach the image edge, and a warped frame has
  made-up pixels there that score as defects.
- Tags that are covered are ignored. `cycles.csv` logs `tag_shift` and
  `tag_note` per brick. `./biddy run --no-tags` switches it all off.

## Layout

| Path | What |
|---|---|
| `biddy` | command-line entry point for every step |
| `station.yaml` | station config: Studio names, policy/anomaly devices, tolerances |
| `cell/run.py` | the loop, run log, mock mode |
| `cell/studio.py` | read-only access to Studio's DB, models and ports |
| `cell/hardware.py` | real and mock arm, cameras, policy |
| `cell/poses.py` | record poses by hand, smooth scripted moves |
| `cell/anomaly.py` | OpenVINO anomaly scoring |
| `cell/inspection.py` | hold detection and the worst-face verdict |
| `cell/tags.py` | AprilTag check before every brick; inspection crops follow the tags |
| `scripts/find_spots.py` | find the arm poses of the inspection spots in the demos |
| `cell/bench.py` | latency of every model on CPU/GPU/NPU |
| `cell/train_policy.py` | train a policy through Studio's API |
| `cell/intel.py` | expose the NPU to OpenVINO |
| `cell/dataset.py` | read Studio's LeRobot datasets |
| `anomaly/` | extract crops, pick ROI, train and export Anomalib |
| `scripts/` | station setup, arm check, dataset check |
| `calibration/` | AprilTag checks that cameras and fixtures haven't moved |
