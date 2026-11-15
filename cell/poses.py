"""Record named arm poses by hand and move between them smoothly.

    python -m cell.poses record              # torque off; move the arm, press Enter per pose
    python -m cell.poses record --only inspect
    python -m cell.poses show
    python -m cell.poses goto home           # torque on; moves slowly to the pose

Poses, in Studio's normalized units (-100..100, gripper 0..100):
    home            parked, out of the top camera's view of the bricks
    inspect         brick held in front of the inspection camera
    good_bowl       gripper over the good bowl (scripted drop)
    defective_bowl  gripper over the defective bowl (scripted drop)
    regrip_above    gripper ~5 cm above a clear spot on the card (scripted regrip)
    regrip          the same spot, fingertips at the table (where the brick is set down)
The gripper value is kept from the recording; the drop opens it separately.

Studio must not be holding the arm (close its robot session first).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

POSE_NAMES = ["home", "inspect", "good_bowl", "defective_bowl", "regrip_above", "regrip"]
GRIPPER = 5


def load(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return {}
    return {k: np.asarray(v, np.float32) for k, v in json.loads(path.read_text()).items()}


def save(path: str | Path, poses: dict[str, np.ndarray]):
    Path(path).write_text(json.dumps({k: [round(float(x), 2) for x in v] for k, v in poses.items()}, indent=2) + "\n")


def body_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest per-joint gap, ignoring the gripper."""
    return float(np.max(np.abs(np.asarray(a)[:GRIPPER] - np.asarray(b)[:GRIPPER])))


def move_to(arm, target: np.ndarray, speed: float = 60.0, hz: float = 50.0, settle: float = 2.0,
            tolerance: float = 3.0, gripper: float | None = None, stop=None) -> bool:
    """Move in a straight joint-space line at `speed` units/s, then wait to settle.

    SO-101 ignores goal_time, so a single far-away command would make it jump at
    full speed; this streams small steps instead. Returns True if the arm got
    within `tolerance` of the target. `stop` is an optional threading.Event.
    """
    target = np.asarray(target, np.float32).copy()
    if gripper is not None:
        target[GRIPPER] = gripper
    start = np.asarray(arm.get_observation().joint_positions, np.float32)
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(start))):
        raise ValueError(f"non-finite joint values: start {start}, target {target}")
    duration = max(float(np.max(np.abs(target - start))) / speed, 0.05)
    t0 = time.monotonic()
    while True:
        if stop is not None and stop.is_set():
            return False
        s = min((time.monotonic() - t0) / duration, 1.0)
        if time.monotonic() - t0 > duration + 5.0:  # never stream forever
            return False
        s = s * s * (3 - 2 * s)  # ease in and out
        arm.send_action(start + (target - start) * s)
        if s >= 1.0:
            break
        time.sleep(1 / hz)
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if stop is not None and stop.is_set():
            return False
        pos = arm.get_observation().joint_positions
        if body_distance(pos, target) <= tolerance:
            return True
        arm.send_action(target)
        time.sleep(1 / hz)
    return False


def _arm(config: str):
    from cell.hardware import build_arm

    cfg = yaml.safe_load(Path(config).read_text())
    return build_arm(cfg), cfg


def cmd_record(args):
    arm, cfg = _arm(args.config)
    path = Path(args.poses or cfg["poses"])
    poses = load(path)
    arm.connect()
    try:
        arm.set_torque(enabled=False)
        print("Torque is off: move the arm by hand (support it, it will drop).")
        for name in args.only or POSE_NAMES:
            input(f"\nMove to '{name}' and press Enter (Ctrl+C to stop)... ")
            pos = np.asarray(arm.get_observation().joint_positions, np.float32)
            poses[name] = pos
            save(path, poses)
            print(f"  {name} = {np.round(pos, 1).tolist()}  -> {path}")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        input("\nHold the arm, then press Enter to turn torque back on at its current position... ")
        # Command the current position before enabling torque so it does not jump.
        arm.send_action(np.asarray(arm.get_observation().joint_positions, np.float32))
        arm.set_torque(enabled=True)
        arm.disconnect()


def cmd_show(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    poses = load(args.poses or cfg["poses"])
    if not poses:
        print("no poses recorded yet")
    for name, v in poses.items():
        print(f"{name:15} {np.round(v, 1).tolist()}")
    missing = [n for n in POSE_NAMES if n not in poses]
    if missing:
        print(f"missing: {missing}")


def cmd_goto(args):
    arm, cfg = _arm(args.config)
    poses = load(args.poses or cfg["poses"])
    if args.name not in poses:
        raise SystemExit(f"no pose {args.name!r}; have {sorted(poses)}")
    arm.connect()
    try:
        ok = move_to(arm, poses[args.name], speed=args.speed)
        print("reached" if ok else "did not settle within tolerance")
    finally:
        arm.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--poses", help="override the poses file from the config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--only", nargs="+", choices=POSE_NAMES)
    r.set_defaults(func=cmd_record)
    sub.add_parser("show").set_defaults(func=cmd_show)
    g = sub.add_parser("goto")
    g.add_argument("name")
    g.add_argument("--speed", type=float, default=40.0, help="units per second")
    g.set_defaults(func=cmd_goto)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
