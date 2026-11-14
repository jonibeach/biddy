"""Find the arm poses where demos hold the brick, to tell inspection holds apart.

    ~/physical-ai-studio/application/backend/.venv/bin/python scripts/find_spots.py [dataset]

Groups every pause in the demos (arm and gripper still, away from home) by
arm pose, lists the poses that come back across many episodes, and saves a
contact sheet (runs/spots.jpg) with the inspection, wrist and top camera at
each. Look at the sheet, then put the poses of each inspection spot under
inspection.views.<view>.poses in station.yaml (the script prints a snippet).
A spot can have several poses if the demos reach it in different ways.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cell.dataset import Dataset, resolve, still_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="latest")
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--tolerance", type=float, default=8.0, help="max joint gap within a group (normalized units)")
    ap.add_argument("--top", type=int, default=10, help="groups to show")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    insp = cfg["inspection"]
    ds = Dataset(resolve(args.dataset, cfg["studio"]["db"]))
    eps = ds.episodes()

    holds = []  # (episode, frame, pose)
    for e in eps:
        for a, b in still_runs(e.state, e.fps, float(insp["still_speed"]), float(insp.get("hold_min_s", 0.3)), gripper=True):
            s = e.state[(a + b) // 2]
            if np.max(np.abs(s[:5] - e.state[0, :5])) > float(insp["home_tolerance"]):
                holds.append((e, (a + b) // 2, s))
    poses = np.array([h[2][:5] for h in holds])
    left, groups = list(range(len(holds))), []
    while left:  # greedy: the pose with the most neighbours founds a group
        d = np.abs(poses[left][:, None] - poses[left][None]).max(-1)
        c = int(np.argmax((d <= args.tolerance).sum(1)))
        members = [left[j] for j in np.flatnonzero(d[c] <= args.tolerance)]
        groups.append(members)
        left = [i for i in left if i not in members]
    groups = sorted(groups, key=len, reverse=True)[: args.top]

    print(f"{len(holds)} pauses in {len(eps)} episodes ({len(holds) / max(1, len(eps)):.1f} per episode)\n")
    rows = []
    for k, members in enumerate(groups):
        n_eps = len({holds[i][0].index for i in members})
        centre = np.median(poses[members], axis=0)
        print(f"g{k}: {len(members):3} pauses in {n_eps:2}/{len(eps)} episodes  pose {[round(float(x), 1) for x in centre]}")
        tiles = []
        for i in [members[j] for j in np.linspace(0, len(members) - 1, 3).astype(int)]:
            e, f, _ = holds[i]
            for cam in cfg["studio"]["cameras"]:
                img = cv2.resize(cv2.cvtColor(ds.frame(int(e.global_index[f]), cam), cv2.COLOR_RGB2BGR), (213, 160))
                cv2.putText(img, f"g{k} {cam} ep{e.index}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                tiles.append(img)
        rows.append(np.hstack(tiles))
    out = Path(cfg["runs_dir"]) / "spots.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 75])
    print(f"\ncontact sheet: {out}")
    print("\nstation.yaml, e.g.:\n  views:\n    side:\n      poses:\n"
          + "".join(f"        - {[round(float(x), 1) for x in np.median(poses[g], axis=0)]}   # g{k}\n"
                    for k, g in enumerate(groups[:2])))


if __name__ == "__main__":
    main()
