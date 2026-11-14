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

1. **Record** in Studio, one task per bowl, e.g. "put the brick in the good bowl"
   and "put the brick in the defective bowl" (they must match `policy.tasks` in
   `station.yaml`). Pause briefly with the brick at the inspection camera in
   every demo. **End the recording session** so Studio writes the dataset.
2. `./biddy check`: flags bad episodes to delete in Studio.
3. `./biddy train-policy --policy smolvla --name smolvla-v1 [--trainer <name>]`:
   trains in Studio and waits; `./biddy status` lists datasets, trainers, jobs, models.
4. `./biddy poses record`: at least `home` and `inspect` (bowls too for scripted drops).
5. Anomaly model from the demos, no extra recording:
   ```bash
   ./biddy extract latest --no-crop     # full frames, to pick the crop box
   ./biddy roi data/anomaly/good/<any>.png
   ./biddy anomaly                      # extract crops -> train -> INT8 -> benchmark
   ```
6. `./biddy mock --cycles 6`, then `./biddy run --cycles 5 --show`.

Steps 2 and 5 together: `./biddy prepare`. Each run writes
`runs/<time>/cycles.csv` (outcome, verdict, score, timings per brick) and the
inspection crops, and prints cycle time, re-grasps and failures.

## How a cycle works

`cell/run.py`: the policy runs with the "good" task until the arm holds still
at the `inspect` pose. The inspection camera's view is compared with the empty
view grabbed at home: no change means nothing was grasped, so the arm goes home
and picks again. Otherwise the Anomalib model scores the crop; borderline
scores get a second look on fresh frames. Then, per `policy.drop`:

- `task_switch` (default): the policy continues with the verdict's task until
  the gripper opens over a bowl.
- `scripted`: the policy stops and the arm replays the recorded bowl pose.

The policy and the inspection share the same camera stream. With
`policy.drop_model` set, one policy picks (e.g. ACT) and another sorts (e.g.
SmolVLA). Every policy action is checked before it reaches the arm: NaN ends
the run, everything else is clamped to the calibrated joint range.

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
| `cell/bench.py` | latency of every model on CPU/GPU/NPU |
| `cell/train_policy.py` | train a policy through Studio's API |
| `cell/intel.py` | expose the NPU to OpenVINO |
| `cell/dataset.py` | read Studio's LeRobot datasets |
| `anomaly/` | extract crops, pick ROI, train and export Anomalib |
| `scripts/` | station setup, arm check, dataset check |
| `calibration/` | AprilTag checks that cameras and fixtures haven't moved |
