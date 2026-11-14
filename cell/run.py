"""Run the inspection cell: pick a brick, inspect it, sort it, repeat.

    PY=~/physical-ai-studio/application/backend/.venv/bin/python
    $PY -m cell.run                      # real arm, newest Studio policy, until Ctrl+C
    $PY -m cell.run --cycles 10 --show   # ten bricks, with a preview window
    $PY -m cell.run --mock --mock-images data/anomaly --cycles 6   # no hardware

One cycle:
  1. pick     The policy runs with the "good" task until the arm holds still at
              the inspect pose (or pick_timeout_s passes).
  2. inspect  The same inspection camera the policy sees is cropped and scored
              by the Anomalib model. If the crop looks like the empty view
              grabbed at home, nothing was grasped: go home and pick again.
              Borderline scores get a second look on fresh frames.
  3. drop     task_switch: the policy carries on with the task for the
              verdict's bowl until the gripper opens. scripted: the policy
              stops and the arm replays the recorded bowl pose.
  4. home     Scripted move to the home pose, ready for the next brick.

Every cycle is logged to runs/<time>/cycles.csv with its inspection crop.
Studio must not be holding the arm or cameras: close its robot session first.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from cell import poses as poses_mod
from cell import studio
from cell.anomaly import AnomalyModel, Verdict, crop, overlay


# --- watching the arm while the policy runs ----------------------------------


class Watcher:
    """RobotRuntime callback that ends a policy run when its phase is done.

    pick: every body joint within `tolerance` of the inspect pose and slower
          than `still_speed` for `settle_ticks` ticks in a row.
    drop: the gripper opened by `release_delta` (the brick was let go) away
          from the inspect pose, then `after_release_s` passed.
    """

    def __init__(self, inspect_pose: np.ndarray, tolerance: float, still_speed: float,
                 settle_ticks: int = 5, release_delta: float = 20.0, after_release_s: float = 0.5):
        self.inspect_pose = np.asarray(inspect_pose, np.float32)
        self.tolerance, self.still_speed = tolerance, still_speed
        self.settle_ticks, self.release_delta, self.after_release_s = settle_ticks, release_delta, after_release_s
        self.stop = None  # set to runtime.stop
        self.begin("pick")

    def begin(self, phase: str, grip_ref: float | None = None):
        self.phase = phase
        self.done = False
        self.still_count = 0
        self.prev: tuple[float, np.ndarray] | None = None
        self.grip_ref = grip_ref
        self.released_at: float | None = None
        self.state: np.ndarray | None = None

    def on_tick(self, event):
        state = np.asarray(event.robot_state.joint_positions, np.float32)
        now = time.monotonic()
        speed = 0.0
        if self.prev is not None and now > self.prev[0]:
            speed = float(np.max(np.abs(state[:5] - self.prev[1][:5]))) / (now - self.prev[0])
        self.prev = (now, state)
        self.state = state
        if self.done:
            return
        near = float(np.max(np.abs(state[:5] - self.inspect_pose[:5]))) <= self.tolerance
        if self.phase == "pick":
            self.still_count = self.still_count + 1 if near and speed <= self.still_speed else 0
            if self.still_count >= self.settle_ticks:
                self._finish()
        elif self.phase == "drop":
            if self.grip_ref is None:
                self.grip_ref = float(state[5])
            if self.released_at is None and not near and state[5] - self.grip_ref >= self.release_delta:
                self.released_at = now
            if self.released_at is not None and now - self.released_at >= self.after_release_s:
                self._finish()

    def _finish(self):
        self.done = True
        if self.stop is not None:
            self.stop()


# --- run log -----------------------------------------------------------------


@dataclass
class Cycle:
    cycle: int
    started: str = ""
    outcome: str = ""            # sorted | no_brick | pick_timeout | drop_timeout | error
    verdict: str = ""            # good | defective
    score: float = float("nan")
    presence: float = float("nan")
    second_look: bool = False
    regrasps: int = 0
    truth: str = ""              # mock runs only: label of the brick image shown
    pick_s: float = 0.0
    inspect_ms: float = 0.0
    drop_s: float = 0.0
    home_s: float = 0.0
    total_s: float = 0.0
    crop: str = ""
    note: str = field(default="")


class RunLog:
    def __init__(self, runs_dir: str | Path, config: dict):
        self.dir = Path(runs_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
        (self.dir / "crops").mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        self.path = self.dir / "cycles.csv"
        self.rows: list[Cycle] = []

    def save_crop(self, c: Cycle, rgb: np.ndarray) -> str:
        name = f"{c.cycle:04d}_{c.verdict or 'none'}_{c.score:.3f}.png"
        cv2.imwrite(str(self.dir / "crops" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return f"crops/{name}"

    def add(self, c: Cycle):
        self.rows.append(c)
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(c)))
            if new:
                w.writeheader()
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(c).items()})

    def summary(self) -> str:
        rows = self.rows
        if not rows:
            return "no cycles"
        sorted_ = [r for r in rows if r.outcome == "sorted"]
        lines = [f"{len(rows)} cycles, {len(sorted_)} sorted, "
                 f"{sum(r.regrasps for r in rows)} re-grasps, "
                 f"{sum(r.outcome != 'sorted' for r in rows)} failed"]
        if sorted_:
            t = np.array([r.total_s for r in sorted_])
            lines.append(f"cycle time median {np.median(t):.1f}s (pick {np.median([r.pick_s for r in sorted_]):.1f}s, "
                         f"inspect {np.median([r.inspect_ms for r in sorted_]):.0f}ms, "
                         f"drop {np.median([r.drop_s for r in sorted_]):.1f}s, home {np.median([r.home_s for r in sorted_]):.1f}s)")
        judged = [r for r in sorted_ if r.truth]
        if judged:
            right = sum(r.truth == r.verdict for r in judged)
            lines.append(f"sorting accuracy {right}/{len(judged)}")
        lines.append(f"log: {self.path}")
        return "\n".join(lines)


# --- the cell ----------------------------------------------------------------


class StopCell(Exception):
    """Stop the cell now: Ctrl+C, or a move that did not get where it had to."""


class KeepOpen:
    """Passes everything to the inference model except close().

    RobotRuntime closes its policy at the end of every run(), but the cell runs
    the same policy many times (pick, drop, every cycle). The cell closes the
    real model itself when it exits.
    """

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def close(self):
        pass


class Cell:
    def __init__(self, cfg: dict, arm, cameras: dict, model, anomaly: AnomalyModel, poses: dict, log: RunLog,
                 show: bool = False):
        from physicalai.runtime import AsyncExecution, PolicySource, RobotRuntime, SyncExecution

        self.cfg, self.arm, self.cameras, self.anomaly, self.poses, self.log = cfg, arm, cameras, anomaly, poses, log
        self.pcfg, self.icfg = cfg["policy"], cfg["inspection"]
        self.show = show
        missing = [p for p in (["home", "inspect"] + (["good_bowl", "defective_bowl"] if self.pcfg["drop"] == "scripted" else []))
                   if p not in poses]
        if missing:
            raise SystemExit(f"poses missing from {cfg['poses']}: {missing}; record them with python -m cell.poses record")
        execution = AsyncExecution() if self.pcfg.get("mode") == "async" else SyncExecution()
        self.model = model
        self.source = PolicySource(KeepOpen(model), execution, task=self.pcfg["tasks"]["good"])
        self.watcher = Watcher(poses["inspect"], self.icfg["pose_tolerance"], self.icfg["still_speed"],
                               release_delta=float(self.pcfg.get("release_delta", 20.0)))
        self.runtime = RobotRuntime(robot=arm, action_source=self.source, fps=float(self.pcfg["fps"]),
                                    cameras=cameras, callbacks=[self.watcher])
        self.watcher.stop = self.runtime.stop
        self.cam = cameras[self.icfg["camera"]]
        self.stop_event = threading.Event()

    # helpers

    def _crops(self, n: int) -> list[np.ndarray]:
        out = []
        for _ in range(n):
            out.append(crop(self.cam.read().data, self.icfg.get("roi")))
        return out

    def _score(self, crops: list[np.ndarray]) -> Verdict:
        verdicts = [self.anomaly(c, with_map=self.show) for c in crops]
        verdicts.sort(key=lambda v: v.score)
        return verdicts[len(verdicts) // 2]  # median

    def _policy(self, phase: str, task: str, timeout: float, grip_ref: float | None = None) -> tuple[bool, float]:
        self.source.set_task(task)
        self.watcher.begin(phase, grip_ref)
        t0 = time.monotonic()
        self.runtime.run(duration_s=timeout, stop_event=self.stop_event)
        # run() swallows Ctrl+C and returns; don't let that look like a timeout.
        if self.runtime.last_run_reason in ("interrupted", "error") or self.stop_event.is_set():
            self.stop_event.set()
            raise StopCell(f"policy run ended: {self.runtime.last_run_reason}")
        return self.watcher.done, time.monotonic() - t0

    def _move(self, pose: np.ndarray, gripper: float | None = None) -> float:
        """Scripted move; raises StopCell if it is stopped or does not arrive."""
        t0 = time.monotonic()
        ok = poses_mod.move_to(self.arm, pose, speed=float(self.cfg["scripted"]["speed"]), gripper=gripper,
                               stop=self.stop_event)
        if not ok:
            raise StopCell("stopped" if self.stop_event.is_set() else
                           f"arm did not reach {np.round(pose, 1).tolist()}; check for a collision or a stuck joint")
        return time.monotonic() - t0

    def _preview(self, rgb: np.ndarray, v: Verdict | None):
        if not self.show:
            return
        img = overlay(rgb, v) if v else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("inspection", img)
        cv2.waitKey(1)

    # one brick

    def cycle(self, i: int) -> Cycle:
        c = Cycle(i, started=datetime.now().isoformat(timespec="seconds"))
        t_start = time.monotonic()
        if hasattr(self.cam, "advance"):  # mock: next brick image
            self.cam.advance()
            c.truth = self.cam.current_path.parent.name if self.cam.current_path else ""
        background = self._crops(1)[0]  # arm is at home: the inspection view is empty
        presence_diff = self.icfg.get("presence_diff")

        for attempt in range(int(self.icfg.get("max_regrasps", 0)) + 1):
            c.regrasps = attempt
            reached, c.pick_s = self._policy("pick", self.pcfg["tasks"]["good"], float(self.pcfg["pick_timeout_s"]))
            if not reached:
                c.outcome, c.note = "pick_timeout", f"no inspect pose after {c.pick_s:.0f}s"
                c.home_s = self._move(self.poses["home"])
                continue
            t0 = time.monotonic()
            crops = self._crops(int(self.icfg.get("frames", 3)))
            mid = crops[len(crops) // 2]
            c.presence = float(np.mean(cv2.absdiff(mid, background)))
            if presence_diff is not None and c.presence < float(presence_diff):
                c.outcome, c.note = "no_brick", f"inspection view unchanged (diff {c.presence:.1f})"
                self._preview(mid, None)
                c.home_s = self._move(self.poses["home"], gripper=float(self.cfg["scripted"]["gripper_open"]))
                if hasattr(self.cam, "regrasp"):  # mock: the next grasp succeeds
                    self.cam.regrasp()
                continue
            v = self._score(crops)
            if abs(v.score - self.anomaly.threshold) < float(self.icfg.get("uncertain_margin", 0.0)):
                c.second_look = True
                v = self._score(crops + self._crops(int(self.icfg.get("frames", 3))))
            c.inspect_ms = (time.monotonic() - t0) * 1000
            c.score, c.verdict = v.score, ("defective" if v.defective else "good")
            c.crop = self.log.save_crop(c, mid)
            self._preview(mid, v)
            break
        else:
            c.total_s = time.monotonic() - t_start
            return c

        if self.pcfg["drop"] == "task_switch":
            grip = float(self.watcher.state[5]) if self.watcher.state is not None else None
            released, c.drop_s = self._policy("drop", self.pcfg["tasks"][c.verdict],
                                              float(self.pcfg["drop_timeout_s"]), grip_ref=grip)
            c.outcome = "sorted" if released else "drop_timeout"
        else:
            t0 = time.monotonic()
            grip = float(self.arm.get_observation().joint_positions[5])
            self._move(self.poses[f"{c.verdict}_bowl"], gripper=grip)  # keep holding the brick
            self._move(self.poses[f"{c.verdict}_bowl"], gripper=float(self.cfg["scripted"]["gripper_open"]))
            c.drop_s = time.monotonic() - t0
            c.outcome = "sorted"
        c.home_s += self._move(self.poses["home"])
        c.total_s = time.monotonic() - t_start
        return c

    def run(self, cycles: int | None):
        with self.runtime:
            self._move(self.poses["home"])
            i = 0
            while (cycles is None or i < cycles) and not self.stop_event.is_set():
                i += 1
                try:
                    c = self.cycle(i)
                except StopCell as e:
                    self.log.add(Cycle(i, outcome="stopped", note=str(e)))
                    print(f"cycle {i}: stopped: {e}")
                    return
                except Exception as e:
                    self.log.add(Cycle(i, outcome="error", note=f"{type(e).__name__}: {e}"))
                    print(f"cycle {i}: ERROR {type(e).__name__}: {e}")
                    raise
                self.log.add(c)
                print(f"cycle {i}: {c.outcome:12} {c.verdict or '-':9} score {c.score:.3f} "
                      f"presence {c.presence:.1f} regrasps {c.regrasps} {c.total_s:.1f}s"
                      + (f" (truth {c.truth})" if c.truth else "") + (f"  {c.note}" if c.note else ""))


# --- setup -------------------------------------------------------------------


def check_cameras(export_dir: Path, cameras: list[str]):
    """Fail early if the policy was trained on different camera names."""
    manifest = json.loads((export_dir / "manifest.json").read_text())
    text = json.dumps(manifest.get("model", {}).get("input_features", manifest))
    trained = {c for c in cameras if f"images.{c}" in text}
    unknown = [c for c in cameras if c not in trained]
    if unknown and trained:
        raise SystemExit(f"policy in {export_dir} was not trained with cameras {unknown}; "
                         f"set studio.cameras in station.yaml to the training cameras")


def build_real(cfg: dict):
    from physicalai.inference import InferenceModel

    from cell.hardware import build_arm, build_cameras

    export_dir = studio.resolve_policy_dir(cfg["policy"]["model"], cfg["studio"]["db"])
    check_cameras(export_dir, cfg["studio"]["cameras"])
    print(f"policy: {export_dir} on {cfg['policy']['device']}")
    model = InferenceModel(export_dir, device=cfg["policy"]["device"])
    return build_arm(cfg), build_cameras(cfg), model


def build_mock(cfg: dict, poses: dict, images: str | None, miss_every: int):
    from cell.hardware import MockArm, MockCamera, MockInspectionCamera, MockPolicy

    arm = MockArm(start=poses["home"])
    cams = {name: MockCamera(name) for name in cfg["studio"]["cameras"]}
    insp = cfg["inspection"]["camera"]
    cams[insp] = MockInspectionCamera(insp, images, arm, poses["inspect"], cfg["inspection"]["pose_tolerance"],
                                      miss_every=miss_every)
    return arm, cams, MockPolicy(poses, cfg["policy"]["tasks"], fps=float(cfg["policy"]["fps"]))


MOCK_POSES = {
    "home": [0, -90, 90, 60, 0, 30],
    "inspect": [30, -20, 30, 10, 0, 5],
    "good_bowl": [-50, -10, 20, 30, 0, 5],
    "defective_bowl": [60, -10, 20, 30, 0, 5],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--cycles", type=int, help="stop after this many bricks (default: until Ctrl+C)")
    ap.add_argument("--drop", choices=["task_switch", "scripted"], help="override policy.drop")
    ap.add_argument("--policy-device", help="override policy.device")
    ap.add_argument("--anomaly-device", help="override inspection.device")
    ap.add_argument("--show", action="store_true", help="preview window with the inspection verdict")
    ap.add_argument("--mock", action="store_true", help="no hardware: mock arm, cameras and policy")
    ap.add_argument("--mock-images", help="folder with good/ and defective/ crops for the mock inspection camera")
    ap.add_argument("--mock-miss-every", type=int, default=0, help="mock: every n-th pick grabs nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.drop:
        cfg["policy"]["drop"] = args.drop
    if args.policy_device:
        cfg["policy"]["device"] = args.policy_device
    if args.anomaly_device:
        cfg["inspection"]["device"] = args.anomaly_device

    poses = poses_mod.load(cfg["poses"])
    if args.mock:
        poses = {k: np.asarray(v, np.float32) for k, v in MOCK_POSES.items()} | poses
        arm, cameras, model = build_mock(cfg, poses, args.mock_images, args.mock_miss_every)
    else:
        arm, cameras, model = build_real(cfg)
    anomaly = AnomalyModel(cfg["inspection"]["model"], cfg["inspection"]["device"])
    print(f"inspection: {cfg['inspection']['model']} on {anomaly.device}, threshold {anomaly.threshold:.3f}")

    log = RunLog(cfg["runs_dir"], cfg)
    cell = Cell(cfg, arm, cameras, model, anomaly, poses, log, show=args.show)
    try:
        cell.run(args.cycles)
    except KeyboardInterrupt:
        cell.stop_event.set()
        print("\nstopped")
    finally:
        close = getattr(model, "close", None)
        if close is not None:
            close()
        print(log.summary())


if __name__ == "__main__":
    main()
