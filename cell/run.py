"""Run the inspection cell: one brick per policy episode, every face inspected.

    ./biddy run                      # real arm, newest Studio policy, until Ctrl+C
    ./biddy run --cycles 10 --show   # ten bricks, with a preview window
    ./biddy mock --cycles 6          # no hardware (mechanics test, synthetic bricks)

The policy plays a whole demo (docs/data_collection.md): hover over the
brick, grasp, show back and bottom to the inspection camera, regrasp, show
both sides, drop in a bowl, return home. While it runs:

  * a RobotRuntime callback spots holds (arm and gripper still, away from home)
    and hands their camera frames to a background inspector;
  * the inspector tells inspection holds (brick in the inspection crop) from
    the hover (scored on the wrist camera), and scores each hold with that
    view's Anomalib model on the NPU;
  * as soon as one face scores at or above its view's threshold, the policy's
    task switches to the defective bowl. The drop comes after the last hold,
    so the switch is in time. No failing face: it stays on the good bowl.

The brick is done when the arm is back home and still. A run with no
inspection hold means nothing was grasped: the policy tries again.

policy.drop picks how the brick reaches the bowl:
  task_switch  one policy does everything (above).
  scripted     the policy stops after inspection.holds_per_brick holds; the arm
               replays the recorded bowl pose, opens, and goes home.
With policy.drop_model set, the pick policy stops after the holds and the
drop policy (e.g. SmolVLA) finishes with the verdict's task.

Every brick is logged to runs/<time>/cycles.csv with its worst crop.
Studio must not be holding the arm or cameras: close its robot session first.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
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
from cell.anomaly import AnomalyModel, overlay
from cell.anomaly import Verdict as Score
from cell.inspection import Verdict, hold_kind, near, presence, split_views, view_crop

# --- inspector: scores holds in the background --------------------------------


class Inspector(threading.Thread):
    """Scores hold frames off the control thread, one hold at a time.

    The watcher sends ("frame", hold_id, state, frames) for a few frames of each
    hold and ("end", hold_id) when the arm moves again. The first frame decides
    what the hold is; its score is the median over its frames. on_defective is
    called once, as soon as a hold's score reaches its view's threshold.
    """

    def __init__(self, models: dict[str, AnomalyModel], views: dict, cfg: dict, on_defective):
        super().__init__(daemon=True)
        self.models, self.views, self.cfg, self.on_defective = models, views, cfg, on_defective
        self.insp_views, self.hover_view = split_views(views)
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.reset(None, None)
        self.start()

    def reset(self, home: np.ndarray | None, backgrounds: dict[str, np.ndarray] | None):
        """New brick: home pose and the empty crop of every inspection view."""
        with self.lock:
            self.home, self.backgrounds = home, backgrounds
            self.verdict = Verdict({v: m.threshold for v, m in self.models.items()}, self.hover_view)
            self.kinds: dict[int, str | None] = {}
            self.samples: dict[int, list[tuple[float, np.ndarray]]] = {}
            self.presence: list[float] = []
            self.worst_crop: tuple[float, str, np.ndarray] | None = None
            self.infer_ms: list[float] = []
            self.switched = False

    def run(self):
        while True:
            msg = self.q.get()
            try:
                if msg[0] == "frame":
                    self._frame(*msg[1:])
                elif msg[0] == "end":
                    self._end(msg[1])
                elif msg[0] == "sync":
                    msg[1].set()
            except Exception as e:  # noqa: BLE001 - never kill the thread; log and carry on
                print(f"inspector: {type(e).__name__}: {e}")

    def sync(self, timeout: float = 5.0):
        """Wait until every message sent so far is handled."""
        done = threading.Event()
        self.q.put(("sync", done))
        done.wait(timeout)

    def _frame(self, hold_id: int, state: np.ndarray, frames: dict):
        with self.lock:
            if self.home is None:
                return
            if hold_id not in self.kinds:
                presences = {v: presence(view_crop(frames[self.views[v]["camera"]], self.views[v]), self.backgrounds[v])
                             for v in self.insp_views}
                seen = any(k not in (None, self.hover_view) for k in self.kinds.values())
                kind = hold_kind(state, self.home, presences, seen, self.cfg, self.hover_view)
                self.kinds[hold_id] = kind if kind in self.models else None
                if kind in presences:
                    self.presence.append(presences[kind])
            kind = self.kinds[hold_id]
            if kind is None:
                return
            rgb = view_crop(frames[self.views[kind]["camera"]], self.views[kind])
        result = self.models[kind](rgb)
        with self.lock:
            self.infer_ms.append(result.latency_ms)
            self.samples.setdefault(hold_id, []).append((result.score, rgb))
            if len(self.samples[hold_id]) >= int(self.cfg.get("frames", 3)):
                self._close(hold_id)

    def _end(self, hold_id: int):
        with self.lock:
            if hold_id in self.samples:
                self._close(hold_id)

    def _close(self, hold_id: int):
        """Called with the lock held: the hold's median goes into the verdict."""
        samples = sorted(self.samples.pop(hold_id), key=lambda s: s[0])
        score, rgb = samples[len(samples) // 2]
        kind = self.kinds[hold_id]
        self.verdict.add(kind, score)
        margin = score - self.verdict.thresholds[kind]
        if self.worst_crop is None or margin > self.worst_crop[0]:
            self.worst_crop = (margin, kind, rgb)
        if margin >= 0 and not self.switched:
            self.switched = True
            self.on_defective()


# --- watcher: RobotRuntime callback ------------------------------------------


class Watcher:
    """Spots holds and decides when a policy run is over.

    end="home":  stop when the arm has left home and is back there, still (the
                 whole demo ran; with no inspection hold it was a missed grasp).
    end="holds": stop once `holds` inspection holds have been scored (the
                 policy hands over to a drop policy or scripted drop).
    """

    # Studio's normalized joint ranges: body joints -100..100, gripper 0..100.
    LOW = np.array([-100, -100, -100, -100, -100, 0], np.float32)
    HIGH = np.full(6, 100, np.float32)

    def __init__(self, inspector: Inspector, cfg: dict, fps: float):
        self.inspector, self.cfg = inspector, cfg
        self.hold_ticks = max(1, int(float(cfg.get("hold_min_s", 0.3)) * fps))
        self.sample_every = max(1, int(fps / 10))
        self.stop = None  # set to runtime.stop
        self.begin("home", None)

    def begin(self, end: str, home: np.ndarray | None, holds: int = 0):
        self.end, self.home, self.holds_wanted = end, home, holds
        self.done = False
        self.prev: tuple[float, np.ndarray] | None = None
        self.still = 0
        self.hold_id = -1
        self.in_hold = False
        self.sent = 0
        self.state: np.ndarray | None = None
        self.home_still = 0
        self.left_home = False

    def on_action_ready(self, *, action, step):
        """Last check before a policy action reaches the arm.

        A NaN/inf action (a broken or untrained export) ends the run instead of
        being sent; everything else is clamped to the calibrated joint range.
        """
        action = np.asarray(action, np.float32)
        if not np.all(np.isfinite(action)):
            raise ValueError(f"policy produced a non-finite action at step {step}: {action}")
        return np.clip(action, self.LOW, self.HIGH)

    def on_tick(self, event):
        state = np.asarray(event.robot_state.joint_positions, np.float32)
        now = time.monotonic()
        speed = 0.0
        if self.prev is not None and now > self.prev[0]:
            speed = float(np.max(np.abs(state - self.prev[1]))) / (now - self.prev[0])  # gripper included
        self.prev = (now, state)
        self.state = state
        if self.done:
            return
        still = speed <= float(self.cfg["still_speed"])
        self.still = self.still + 1 if still else 0

        if self.in_hold and not still:
            self.in_hold = False
            self.inspector.q.put(("end", self.hold_id))
        elif not self.in_hold and self.still >= self.hold_ticks:
            self.in_hold = True
            self.hold_id += 1
            self.sent = 0
        if (self.in_hold and event.camera_frames and self.sent < int(self.cfg.get("frames", 3))
                and self.still % self.sample_every == 0):
            frames = {name: f.data for name, f in event.camera_frames.items()}
            self.inspector.q.put(("frame", self.hold_id, state, frames))
            self.sent += 1

        if self.end == "home":
            at_home = self.home is not None and near(state, self.home, float(self.cfg["home_tolerance"]))
            self.left_home = self.left_home or not at_home
            self.home_still = self.home_still + 1 if at_home and still else 0
            if self.left_home and self.home_still >= self.hold_ticks:
                self._finish()
        elif self.end == "holds" and self.inspector.verdict.holds >= self.holds_wanted:
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
    outcome: str = ""            # sorted | no_brick | timeout | stopped | error
    verdict: str = ""            # good | defective
    worst_view: str = ""         # which camera saw the worst face
    score: float = float("nan")  # worst face's score
    threshold: float = float("nan")
    uncertain: bool = False      # worst face within uncertain_margin of its threshold
    holds: int = 0               # inspection holds scored
    hover_holds: int = 0
    switched: bool = False       # task switched to the defective bowl during the run
    presence: float = float("nan")
    regrasps: int = 0
    truth: str = ""              # mock runs only
    infer_ms: float = float("nan")  # median anomaly inference per frame
    episode_s: float = 0.0
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
        name = f"{c.cycle:04d}_{c.verdict or 'none'}_{c.worst_view}_{c.score:.3f}.png"
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
        done = [r for r in rows if r.outcome == "sorted"]
        lines = [f"{len(rows)} bricks, {len(done)} sorted, {sum(r.regrasps for r in rows)} re-grasps, "
                 f"{sum(r.outcome != 'sorted' for r in rows)} failed, {sum(r.uncertain for r in done)} uncertain"]
        if done:
            lines.append(f"cycle time median {np.median([r.total_s for r in done]):.1f}s, "
                         f"anomaly inference median {np.nanmedian([r.infer_ms for r in done]):.1f}ms/frame")
        judged = [r for r in done if r.truth]
        if judged:
            lines.append(f"sorting accuracy {sum(r.truth == r.verdict for r in judged)}/{len(judged)}")
        lines.append(f"log: {self.path}")
        return "\n".join(lines)


# --- the cell ----------------------------------------------------------------


class StopCell(Exception):
    """Stop the cell now: Ctrl+C, or a move that did not get where it had to."""


class SwitchSource:
    """One RobotRuntime action source that hands over to the policy for the phase.

    RobotRuntime owns the arm and cameras and takes a single action source;
    the cell sets `current` before each run() so pick and drop can use
    different policies (e.g. ACT to pick and inspect, SmolVLA to sort).
    """

    def __init__(self, sources: dict):
        self.sources = sources
        self.current = next(iter(sources.values()))

    def use(self, name: str):
        self.current = self.sources[name]

    def connect(self, *, bus, session_id):
        self.current.connect(bus=bus, session_id=session_id)

    def update(self, robot_state, camera_frames, step):
        return self.current.update(robot_state, camera_frames, step)

    def disconnect(self):
        self.current.disconnect()

    @property
    def action_queue(self):
        return self.current.action_queue


class KeepOpen:
    """Passes everything to the inference model except close().

    RobotRuntime closes its policy at the end of every run(), but the cell runs
    the same policy for every brick. The cell closes the real model at exit.
    """

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def close(self):
        pass


class Cell:
    def __init__(self, cfg: dict, arm, cameras: dict, model, anomaly: dict[str, AnomalyModel], poses: dict,
                 log: RunLog, show: bool = False, drop_model=None):
        from physicalai.runtime import AsyncExecution, PolicySource, RobotRuntime, SyncExecution

        self.cfg, self.arm, self.cameras, self.poses, self.log = cfg, arm, cameras, poses, log
        self.pcfg, self.icfg = cfg["policy"], cfg["inspection"]
        self.show = show
        self.mode = "handover" if drop_model is not None else self.pcfg["drop"]
        needed = ["home"] + (["good_bowl", "defective_bowl"] if self.mode == "scripted" else [])
        missing = [p for p in needed if p not in poses]
        if missing:
            raise SystemExit(f"poses missing from {cfg['poses']}: {missing}; record them with ./biddy poses record")
        self.insp_views, self.hover_view = split_views(self.icfg["views"])
        if not any(v in anomaly for v in self.insp_views):
            raise SystemExit(f"no anomaly model for any inspection view {self.insp_views}; train with ./biddy anomaly")

        def source(m):
            execution = AsyncExecution() if self.pcfg.get("mode") == "async" else SyncExecution()
            return PolicySource(KeepOpen(m), execution, task=self.pick_task)

        self.tasks = self.pcfg["tasks"]
        self.pick_task = self.tasks.get("pick") or self.tasks["good"]
        self.models = [model] + ([drop_model] if drop_model is not None else [])
        pick_source = source(model)
        self.source = SwitchSource({"pick": pick_source,
                                    "drop": source(drop_model) if drop_model is not None else pick_source})
        self.inspector = Inspector(anomaly, self.icfg["views"], self.icfg, self._on_defective)
        self.watcher = Watcher(self.inspector, self.icfg, float(self.pcfg["fps"]))
        self.runtime = RobotRuntime(robot=arm, action_source=self.source, fps=float(self.pcfg["fps"]),
                                    cameras=cameras, callbacks=[self.watcher])
        self.watcher.stop = self.runtime.stop
        self.insp_cam = cameras[self.icfg["views"][self.insp_views[0]]["camera"]]
        self.stop_event = threading.Event()

    def _on_defective(self):
        """Inspector thread: a face failed. In task_switch mode, steer to the defective bowl now."""
        if self.mode == "task_switch":
            self.source.current.set_task(self.tasks["defective"])

    def _policy(self, phase: str, task: str, timeout: float, end: str, holds: int = 0) -> tuple[bool, float]:
        self.source.use(phase)
        self.source.current.set_task(task)
        self.watcher.begin(end, self.poses["home"], holds)
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

    def _preview(self, rgb: np.ndarray, verdict: str, score: float):
        if not self.show:
            return
        img = overlay(rgb, Score(score, verdict == "defective", 0.0))
        cv2.imshow("worst face", img)
        cv2.waitKey(1)

    # one brick

    def cycle(self, i: int) -> Cycle:
        c = Cycle(i, started=datetime.now().isoformat(timespec="seconds"))
        t_start = time.monotonic()
        if hasattr(self.insp_cam, "next_brick"):  # mock: put the next brick on the table
            c.truth = self.insp_cam.next_brick()
        views = self.icfg["views"]
        timeout = float(self.pcfg["episode_timeout_s"])
        holds_wanted = int(self.icfg.get("holds_per_brick", 4))

        for attempt in range(int(self.icfg.get("max_regrasps", 0)) + 1):
            c.regrasps = attempt
            # Arm at home: the inspection view is empty; that is the presence reference.
            frames = {cam: self.cameras[cam].read().data for cam in {views[v]["camera"] for v in self.insp_views}}
            self.inspector.reset(self.poses["home"], {v: view_crop(frames[views[v]["camera"]], views[v])
                                                      for v in self.insp_views})
            if self.mode == "task_switch":
                done, c.episode_s = self._policy("pick", self.pick_task, timeout, "home")
            else:
                done, c.episode_s = self._policy("pick", self.pick_task, timeout, "holds", holds_wanted)
            self.inspector.sync()
            v = self.inspector.verdict
            if v.holds == 0:
                c.outcome, c.note = "no_brick", "no inspection hold showed a brick"
                c.home_s = self._move(self.poses["home"], gripper=float(self.cfg["scripted"]["gripper_open"]))
                if hasattr(self.insp_cam, "regrasp"):
                    self.insp_cam.regrasp()
                continue
            break
        else:
            c.total_s = time.monotonic() - t_start
            return c

        with self.inspector.lock:
            v = self.inspector.verdict
            c.verdict = "defective" if v.defective else "good"
            c.worst_view, c.score = v.worst()
            c.threshold = v.thresholds.get(c.worst_view, float("nan"))
            c.uncertain = v.margin() < float(self.icfg.get("uncertain_margin", 0.0))
            c.holds, c.hover_holds = v.holds, len(v.scores.get(self.hover_view, [])) if self.hover_view else 0
            c.switched = self.inspector.switched and self.mode == "task_switch"
            c.presence = float(np.median(self.inspector.presence)) if self.inspector.presence else float("nan")
            c.infer_ms = float(np.median(self.inspector.infer_ms)) if self.inspector.infer_ms else float("nan")
            worst = self.inspector.worst_crop
        if worst is not None:
            c.crop = self.log.save_crop(c, worst[2])
            self._preview(worst[2], c.verdict, c.score)

        if self.mode == "task_switch":
            c.outcome = "sorted" if done else "timeout"
            if not done:
                c.note = f"not back home after {timeout:.0f}s"
                c.home_s = self._move(self.poses["home"])
        elif self.mode == "handover":
            if not done:
                c.outcome, c.note = "timeout", f"{c.holds} of {holds_wanted} inspection holds in {timeout:.0f}s"
            t0 = time.monotonic()
            done, _ = self._policy("drop", self.tasks[c.verdict], timeout, "home")
            c.drop_s = time.monotonic() - t0
            c.outcome = "sorted" if done else "timeout"
            if not done:
                c.home_s = self._move(self.poses["home"])
        else:  # scripted
            if not done:
                c.note = f"{c.holds} of {holds_wanted} inspection holds in {timeout:.0f}s; sorting on those"
            t0 = time.monotonic()
            grip = float(self.arm.get_observation().joint_positions[5])
            bowl = self.poses[f"{c.verdict}_bowl"]
            self._move(bowl, gripper=grip)  # keep holding the brick
            self._move(bowl, gripper=float(self.cfg["scripted"]["gripper_open"]))
            c.drop_s = time.monotonic() - t0
            c.home_s = self._move(self.poses["home"])
            c.outcome = "sorted"
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
                    print(f"brick {i}: stopped: {e}")
                    return
                except Exception as e:
                    self.log.add(Cycle(i, outcome="error", note=f"{type(e).__name__}: {e}"))
                    print(f"brick {i}: ERROR {type(e).__name__}: {e}")
                    raise
                self.log.add(c)
                print(f"brick {i}: {c.outcome:9} {c.verdict or '-':9} worst {c.worst_view or '-'} {c.score:.3f}"
                      f"/{c.threshold:.3f}{' UNCERTAIN' if c.uncertain else ''} holds {c.hover_holds}+{c.holds} "
                      f"regrasps {c.regrasps} {c.total_s:.1f}s"
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


def load_policy(export_dir: Path, cfg: dict, role: str = "pick"):
    """InferenceModel on the configured device, falling back to FP32 if FP16 gives NaN.

    OpenVINO runs GPU models in FP16 by default. A joint that never moved in
    the demos has a standard deviation of ~0, and its normalization then
    divides by zero in FP16 (all-NaN actions). policy.precision: auto probes
    once and reloads in FP32 (about 2x slower) only when needed.
    """
    from physicalai.inference import InferenceModel

    from cell.bench import policy_obs

    device, precision = cfg["policy"]["device"], str(cfg["policy"].get("precision", "auto")).lower()
    kwargs = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "f32" else {}
    model = InferenceModel(export_dir, device=device, **kwargs)
    if precision == "auto" and device.upper() != "CPU":
        if not np.all(np.isfinite(model.predict_action_chunk(policy_obs(Path(export_dir), cfg["policy"]["tasks"]["good"])))):
            print(f"{role} policy gives NaN in FP16 on {device} (a joint that never moves in the demos?); using FP32")
            model.close()
            model = InferenceModel(export_dir, device=device, INFERENCE_PRECISION_HINT="f32")
        else:
            model.reset()
    print(f"{role} policy: {export_dir} on {device}")
    return model


def load_anomaly(cfg: dict) -> dict[str, AnomalyModel]:
    """One model per inspection view whose model folder exists."""
    models = {}
    for view, v in cfg["inspection"]["views"].items():
        if (Path(v["model"]) / "model.xml").exists():
            models[view] = AnomalyModel(v["model"], cfg["inspection"]["device"])
            print(f"{view} view: {v['model']} on {models[view].device}, threshold {models[view].threshold:.3f}")
        else:
            print(f"{view} view: no model at {v['model']}, not inspected")
    return models


def build_real(cfg: dict):
    from cell.hardware import build_arm, build_cameras

    def load(spec, role):
        export_dir = studio.resolve_policy_dir(spec, cfg["studio"]["db"])
        check_cameras(export_dir, cfg["studio"]["cameras"])
        return load_policy(export_dir, cfg, role)

    model = load(cfg["policy"]["model"], "pick")
    drop = cfg["policy"].get("drop_model")
    drop_model = load(drop, "drop") if drop else None
    return build_arm(cfg), build_cameras(cfg), model, drop_model


def build_mock(cfg: dict, poses: dict, miss_every: int):
    from cell.hardware import MockArm, MockCamera, MockPolicy, MockStation

    arm = MockArm(start=poses["home"])
    views = cfg["inspection"]["views"]
    cams = {name: MockCamera(name) for name in cfg["studio"]["cameras"]}
    station = MockStation(arm, poses, views, miss_every=miss_every)
    for view in views:
        cams[views[view]["camera"]] = station.camera(view)
    return arm, cams, MockPolicy(poses, cfg["policy"]["tasks"], fps=float(cfg["policy"]["fps"])), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--cycles", type=int, help="stop after this many bricks (default: until Ctrl+C)")
    ap.add_argument("--drop", choices=["task_switch", "scripted"], help="override policy.drop")
    ap.add_argument("--policy-device", help="override policy.device")
    ap.add_argument("--anomaly-device", help="override inspection.device")
    ap.add_argument("--show", action="store_true", help="preview window with each brick's worst face")
    ap.add_argument("--mock", action="store_true", help="no hardware: mock arm, cameras and policy")
    ap.add_argument("--mock-miss-every", type=int, default=0, help="mock: every n-th pick grabs nothing")
    ap.add_argument("--mock-policy", help="mock: drive the mock arm with this exported policy instead of the scripted one")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.drop:
        cfg["policy"]["drop"] = args.drop
    if args.policy_device:
        cfg["policy"]["device"] = args.policy_device
    if args.anomaly_device:
        cfg["inspection"]["device"] = args.anomaly_device

    poses = {k: np.asarray(v, np.float32) for k, v in poses_mod.load(cfg["poses"]).items()}
    if args.mock:
        from cell.hardware import MOCK_POSES

        poses = {k: np.asarray(v, np.float32) for k, v in MOCK_POSES.items()}  # the mock policy's own route
        arm, cameras, model, drop_model = build_mock(cfg, poses, args.mock_miss_every)
        if args.mock_policy:  # real exported policy (InferenceModel) driving the mock arm
            check_cameras(Path(args.mock_policy), cfg["studio"]["cameras"])
            model = load_policy(Path(args.mock_policy), cfg)
    else:
        arm, cameras, model, drop_model = build_real(cfg)
    anomaly = load_anomaly(cfg)

    log = RunLog(cfg["runs_dir"], cfg)
    cell = Cell(cfg, arm, cameras, model, anomaly, poses, log, show=args.show, drop_model=drop_model)
    try:
        cell.run(args.cycles)
    except KeyboardInterrupt:
        cell.stop_event.set()
        print("\nstopped")
    finally:
        for m in cell.models:
            close = getattr(m, "close", None)
            if close is not None:
                close()
        print(log.summary())


if __name__ == "__main__":
    main()
