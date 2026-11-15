"""Scripted inspection of a hand-loaded brick: collect anomaly crops from fixed poses.

    PY=~/physical-ai-studio/application/backend/.venv/bin/python
    $PY scripts/inspect_loaded.py good            # one brick: load it, it is shown and saved
    $PY scripts/inspect_loaded.py defective --bricks 3 --wait 6

Why: crops from policy grasps vary with every grasp (studs, tilt, jaw tips),
and the policy misses often, so anomaly data from live runs came in slowly and
inconsistently. Here the brick is put deep and square between the jaws by hand,
and the arm shows it at the recorded inspection poses (station.yaml
inspection.views.<view>.poses), so every crop has the same framing the cell
should present.

Per brick:
  1. the arm goes to the load pose (the first side pose: wrist level, in reach)
     and opens the gripper; you have --wait seconds to put the brick in;
  2. it closes, then holds still at each pose and saves --frames crops per pose
     (brick found by cell/brickcrop.py against the empty view captured at home);
  3. it goes back to the load pose and opens: take the brick out.
Crops land in data/live/<label>/<view>/load_<time>_<pose>_<i>.png, with the
current model's score in the name (e.g. ..._0.981.png).
Studio must not hold the arm (close its UI tab).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cell import poses as poses_mod  # noqa: E402
from cell.hardware import build_arm, build_cameras  # noqa: E402
from cell.inspection import score_crop, split_views  # noqa: E402

GRIP_OPEN = 60.0
GRIP_CLOSED = 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", choices=["good", "defective"])
    ap.add_argument("--bricks", type=int, default=1, help="bricks to do in a row")
    ap.add_argument("--wait", type=float, default=6.0, help="seconds to load the brick")
    ap.add_argument("--frames", type=int, default=3, help="crops per pose")
    ap.add_argument("--settle", type=float, default=0.8, help="seconds still before the first frame")
    ap.add_argument("--speed", type=float, default=50.0, help="units/s for the moves")
    ap.add_argument("--no-score", action="store_true", help="don't load the anomaly models")
    ap.add_argument("--config", default=str(ROOT / "station.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    icfg = cfg["inspection"]
    views, _ = split_views(icfg["views"])
    home = poses_mod.load(ROOT / cfg["poses"])["home"]
    stops = [(v, i, np.asarray(p, np.float32)) for v in views for i, p in enumerate(icfg["views"][v].get("poses") or [])]
    if not stops:
        raise SystemExit("no inspection poses in station.yaml")
    # Long-brick poses roll the wrist ~50: skip them for 2x2 bricks (one pose per roll sign is enough).
    stops = [s for s in stops if abs(float(s[2][4])) < 20]
    stops = [s for n, s in enumerate(stops) if sum(1 for t in stops[:n] if t[0] == s[0]) < 2]  # first 2 per view
    load = stops[0][2]
    # The load pose is the first side pose, so a hand may still be in view there:
    # visit the other poses first and come back to it last.
    stops = stops[1:] + stops[:1]

    models = {}
    if not args.no_score:
        from cell.run import load_anomaly

        models = load_anomaly(cfg)

    arm = build_arm(cfg)
    cams = build_cameras(cfg)
    arm.connect()
    for c in cams.values():
        c.connect()
    try:
        def pose6(p5, grip):
            return np.concatenate([p5[:5], [grip]]).astype(np.float32)

        print("going home for the empty views...")
        poses_mod.move_to(arm, home, speed=args.speed)
        # Let auto-exposure settle after connecting: an empty view grabbed too early
        # differs everywhere from later frames, and the "brick" became the whole view.
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            for c in cams.values():
                c.read()
        empty = {cam: cams[cam].read().data for cam in {icfg["views"][v]["camera"] for v in views}}
        for b in range(args.bricks):
            poses_mod.move_to(arm, pose6(load, GRIP_OPEN), speed=args.speed)
            for s in range(int(args.wait), 0, -1):
                print(f"brick {b + 1}/{args.bricks} ({args.label}): put it deep and square between the jaws... {s}",
                      flush=True)
                time.sleep(1)
            poses_mod.move_to(arm, pose6(load, GRIP_CLOSED), speed=args.speed, settle=0.5)
            # The gripper reading can't tell a held brick from an empty grip on this arm
            # (1.9 with a brick, ~1.2 empty); an empty grip shows up as "no brick found" below.
            stamp = time.strftime("%H%M%S")
            for view, i, p in stops:
                poses_mod.move_to(arm, pose6(p, GRIP_CLOSED), speed=args.speed, settle=args.settle)
                cam = icfg["views"][view]["camera"]
                saved = []
                for k in range(args.frames):
                    rgb = score_crop(cams[cam].read().data, icfg["views"][view], empty.get(cam))
                    if rgb is not None and rgb.shape[0] > 300:  # a "brick" this big is the whole view changing
                        print(f"  {view} pose {i}: crop {rgb.shape[0]} px, the empty view no longer matches; skipped")
                        rgb = None
                        continue
                    if rgb is None:
                        print(f"  {view} pose {i}: no brick found in frame {k}")
                        continue
                    score = models[view](rgb).score if view in models else float("nan")
                    out = ROOT / "data/live" / args.label / view
                    out.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out / f"load_{stamp}_p{i}_{k}_{score:.3f}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    saved.append(score)
                    time.sleep(0.15)
                print(f"  {view} pose {i}: {len(saved)} crops, scores " + " ".join(f"{s:.3f}" for s in saved))
            poses_mod.move_to(arm, pose6(load, GRIP_CLOSED), speed=args.speed, settle=0.3)
            poses_mod.move_to(arm, pose6(load, GRIP_OPEN), speed=args.speed, settle=0.5)
            print("  done: take the brick out")
            time.sleep(2.0)
        poses_mod.move_to(arm, home, speed=args.speed)
    finally:
        for c in cams.values():
            c.disconnect()
        arm.disconnect()


if __name__ == "__main__":
    main()
