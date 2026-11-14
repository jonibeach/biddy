# Biddy: autonomous brick inspection cell

Intel Physical AI Challenge, Junction 2026. An SO-101 arm picks each Duplo
brick, shows it to the inspection camera and drops it into the good or
defective bowl. See [AGENTS.md](AGENTS.md) for the task and hardware.

## Two Python environments

| Env | Path | Used for |
|---|---|---|
| Studio venv | `~/physical-ai-studio/application/backend/.venv` | the cell, poses, dataset tools (has physicalai, LeRobot, OpenVINO) |
| Repo venv | `.venv` (`uv sync --group anomaly`) | Anomalib training only |

```bash
PY=~/physical-ai-studio/application/backend/.venv/bin/python
```

Robot calibration, serial numbers and camera identities come from Studio's
database (`station.yaml` only names them), so the cell drives exactly what the
demos were recorded with. Check what it finds: `$PY -m cell.studio`.

**Close Studio's robot session before running anything that touches the arm
or cameras.** Studio locks them while a session is open.

## From recorded demos to a running cell

1. **Record** in Studio. Use one task per bowl, e.g. "put the brick in the good bowl"
   and "put the brick in the defective bowl" (these must match `policy.tasks` in
   `station.yaml`). Pause briefly with the brick in front of the inspection
   camera in every demo. **End the recording session** so Studio writes the
   dataset to disk.
2. **Check the dataset** and delete flagged episodes in Studio:
   `$PY scripts/check_dataset.py latest`
3. **Train the policy** in Studio (SmolVLA reads the task text, which the cell
   uses to pick the bowl). Make sure the OpenVINO export is on.
4. **Record poses**, at least `home` and `inspect`. Add the bowls too if you
   want scripted drops:
   `$PY -m cell.poses record`, then `$PY -m cell.poses show`
5. **Anomaly data from the demos**. No extra recording needed:
   ```bash
   $PY anomaly/extract.py latest --no-crop        # full frames, to pick the crop box
   $PY anomaly/roi.py data/anomaly/good/<any>.png  # drag a box around the held brick
   $PY anomaly/extract.py latest                   # cropped, labeled by task text
   ```
6. **Train the anomaly model** (repo venv):
   ```bash
   uv run --group anomaly python anomaly/train.py --data data/anomaly --out models/anomaly --int8
   $PY -m cell.anomaly models/anomaly-int8 --bench   # latency per device
   ```
7. **Dry run without hardware**, then for real:
   ```bash
   $PY -m cell.run --mock --mock-images data/anomaly --cycles 6
   $PY -m cell.run --cycles 5 --show
   ```

Each run writes `runs/<time>/cycles.csv` (outcome, verdict, score, timings per
brick) and the inspection crops, and prints cycle time and failure counts.

## How a cycle works

`cell/run.py`: the policy runs with the "good" task until the arm holds still
at the `inspect` pose. The inspection camera's view is compared with the empty
view grabbed at home: no change means nothing was grasped, so the arm goes home
and picks again. Otherwise the Anomalib model scores the crop; borderline
scores get a second look on fresh frames. Then, per `policy.drop`:

- `task_switch` (default): the policy continues with the verdict's task until
  the gripper opens over a bowl.
- `scripted`: the policy stops and the arm replays the recorded bowl pose.

The policy and the inspection share the same camera stream.

## Layout

| Path | What |
|---|---|
| `station.yaml` | station config: Studio names, policy/anomaly devices, tolerances |
| `cell/run.py` | the loop, run log, mock mode |
| `cell/studio.py` | read-only access to Studio's DB, models and ports |
| `cell/hardware.py` | real and mock arm, cameras, policy |
| `cell/poses.py` | record poses by hand, smooth scripted moves |
| `cell/anomaly.py` | OpenVINO anomaly scoring and benchmark |
| `cell/dataset.py` | read Studio's LeRobot datasets |
| `anomaly/` | extract crops, pick ROI, train and export Anomalib |
| `scripts/` | station setup, arm check, dataset check |
| `calibration/` | AprilTag checks that cameras and fixtures haven't moved |
