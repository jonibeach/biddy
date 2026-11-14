"""Sanity-check a Studio dataset before uploading it for training.

    ~/physical-ai-studio/application/backend/.venv/bin/python scripts/check_dataset.py latest
    ... scripts/check_dataset.py pick-inspect-sort        # Studio dataset name
    ... scripts/check_dataset.py path/to/dataset

Flags episodes worth deleting in Studio: very short or long ones, ones where
the gripper never closes or the brick never pauses at the inspection camera,
and teleop glitches (sudden action jumps). Also checks camera names against
station.yaml and how the tasks are balanced.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cell import poses as poses_mod  # noqa: E402
from cell.dataset import Dataset, resolve  # noqa: E402
from cell.inspection import INSPECTION, WRIST, find_holds, view_crop  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="latest")
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--jump", type=float, default=25.0, help="flag action jumps larger than this per frame")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = resolve(args.dataset, cfg["studio"]["db"])
    ds = Dataset(root)
    eps = ds.episodes()
    print(f"{root}\n{len(eps)} episodes, {len(ds.frames)} frames at {ds.fps:g} fps\n")

    want = {f"observation.images.{c}" for c in cfg["studio"]["cameras"]}
    have = set(ds.camera_keys)
    if want != have:
        print(f"CAMERAS DIFFER from station.yaml: missing {sorted(want - have)}, extra {sorted(have - want)}\n")

    home = poses_mod.load(cfg["poses"]).get("home")
    insp = cfg["inspection"]
    view = insp["views"][INSPECTION]
    expected = int(cfg.get("scripted", {}).get("after_holds", 4))
    lengths = np.array([e.seconds for e in eps])
    med = float(np.median(lengths)) if len(eps) else 0.0
    flagged = []
    print(f"{'ep':>4} {'sec':>6} {'grip':>11} {'jump':>5} {'holds':>6}  task / flags")
    for e in eps:
        flags = []
        g = e.state[:, 5]
        grip_range = float(g.max() - g.min())
        jump = float(np.abs(np.diff(e.action, axis=0)).max()) if len(e.action) > 1 else 0.0
        holds = find_holds(e.state, e.fps, lambda i, e=e: view_crop(ds.frame(int(e.global_index[i]), view["camera"]), view),
                           insp, home)
        n_insp = sum(h.kind == INSPECTION for h in holds)
        n_wrist = sum(h.kind == WRIST for h in holds)
        if e.seconds < 0.5 * med:
            flags.append("short")
        if e.seconds > 2.0 * med:
            flags.append("long")
        if grip_range < 10:
            flags.append("gripper barely moves")
        if jump > args.jump:
            flags.append(f"action jump {jump:.0f}")
        if n_insp < expected:
            flags.append(f"{n_insp} inspection holds, protocol has {expected}")
        if n_wrist == 0:
            flags.append("no hover hold before the grasp")
        if "|" in e.task:
            flags.append("several tasks in one episode")
        if flags:
            flagged.append(e.index)
        print(f"{e.index:>4} {e.seconds:>6.1f} {g.min():>5.0f}-{g.max():<5.0f} {jump:>5.0f} {n_wrist:>2}+{n_insp:<3}  "
              f"{e.task}{'  <-- ' + ', '.join(flags) if flags else ''}")

    print("\ntasks:")
    for task, n in Counter(e.task for e in eps).most_common():
        print(f"  {n:>3}  {task}")
    tasks_cfg = cfg["policy"]["tasks"]
    known = {e.task for e in eps}
    for k in ("good", "defective"):
        if tasks_cfg[k] not in known:
            print(f"  note: station.yaml task {k!r} = {tasks_cfg[k]!r} is not in this dataset")
    if eps:
        all_state = np.concatenate([e.state for e in eps])
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for name, std in zip(names, all_state.std(axis=0)):
            if std < 0.5:
                print(f"note: {name} barely moves in the demos (std {std:.2f}); the policy may give NaN in FP16 "
                      "on the GPU. The cell falls back to FP32 automatically (policy.precision: auto).")
    print(f"\nlength median {med:.1f}s; flagged {len(flagged)}: {flagged}")
    print("(holds: wrist hover + inspection holds; set inspection.views.inspection.roi for reliable detection)")


if __name__ == "__main__":
    main()
