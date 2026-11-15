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
from cell import tags as tags_mod
from cell.anomaly import AnomalyModel, overlay
from cell.anomaly import Verdict as Score
from cell.inspection import Verdict, at_pose, hold_kind, near, presence, score_crop, split_views, view_crop

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
        self.models, self.station_views, self.cfg, self.on_defective = models, views, cfg, on_defective
        self.insp_views, self.hover_view = split_views(views)
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.save_dir: Path | None = None  # --collect: data/live/<label>; each scored crop is saved
        self.reset(None, None)
        self.start()

    def reset(self, home: np.ndarray | None, backgrounds: dict[str, np.ndarray] | None, views: dict | None = None,
              empty_frames: dict[str, np.ndarray] | None = None):
        """New brick: home pose and the empty crop of every inspection view.

        views: this brick's views when a camera moved and its crop boxes
        follow the AprilTags (cell/tags.py); default: the station's.
        empty_frames: each camera's full frame with the arm at home, which
        `crop: brick` views compare against to find the brick (cell/brickcrop.py).
        """
        with self.lock:
            self.home, self.backgrounds, self.views = home, backgrounds, views or self.station_views
            self.empty_frames = empty_frames or {}
            self.verdict = Verdict({v: m.threshold for v, m in self.models.items()}, self.hover_view)
            self.kinds: dict[int, str | None] = {}
            self.samples: dict[int, list[tuple[float, np.ndarray]]] = {}
            self.presence: list[float] = []
            self.worst_crop: tuple[float, str, np.ndarray] | None = None
            self.infer_ms: list[float] = []
            self.switched = False
            self.save_prefix = time.strftime("%H%M%S")
            self.grip_max = 0.0  # widest gripper seen this brick: a drop from it means the grasp happened

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
            self.grip_max = max(self.grip_max, float(state[5]))
            if hold_id not in self.kinds:
                presences = {v: presence(view_crop(frames[self.views[v]["camera"]], self.views[v]), self.backgrounds[v])
                             for v in self.insp_views}
                # The hover (wrist camera over the brick) comes before the grasp. Once the
                # gripper has closed well below its widest, a pause on the way to a spot
                # is not the hover: scoring it on the wrist model once sent a good brick
                # to the defective bowl.
                grasped = self.grip_max - float(state[5]) > float(self.cfg.get("grasp_drop", 8.0))
                seen = grasped or any(k not in (None, self.hover_view) for k in self.kinds.values())
                kind = hold_kind(state, self.home, presences, seen, self.cfg, self.hover_view)
                self.kinds[hold_id] = kind if kind in self.models else None
                if kind in presences:
                    self.presence.append(presences[kind])
                dist = {v: at_pose(state, self.views[v], float(self.cfg.get("pose_tolerance", 10.0)))
                        for v in self.insp_views}
                print(f"  hold {hold_id}: {kind or '-'}  grip {state[5]:.1f}/{self.grip_max:.1f}"
                      f"{' grasped' if grasped else ''}  presence "
                      + " ".join(f"{v} {p:.1f}" for v, p in presences.items())
                      + "  pose " + " ".join(f"{v} {'-' if d is None else f'{d:.1f}'}" for v, d in dist.items())
                      + "  joints " + " ".join(f"{x:.0f}" for x in state[:5]), flush=True)
            kind = self.kinds[hold_id]
            if kind is None:
                return
            cam = self.views[kind]["camera"]
            rgb = score_crop(frames[cam], self.views[kind], self.empty_frames.get(cam))
            if rgb is None:  # crop: brick found no brick in this frame
                print(f"  hold {hold_id}: {kind} - no brick found in the crop", flush=True)
                return
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
        if self.save_dir is not None:  # --collect: every scored crop, for retraining on live grips
            d = self.save_dir / kind
            d.mkdir(parents=True, exist_ok=True)
            for i, (sc, im) in enumerate(samples):
                cv2.imwrite(str(d / f"{self.save_prefix}_h{hold_id:02d}_{i}_{sc:.3f}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        self.verdict.add(kind, score)
        margin = score - self.verdict.thresholds[kind]
        if self.worst_crop is None or margin > self.worst_crop[0]:
            self.worst_crop = (margin, kind, rgb)
        if margin >= 0 and not self.switched:
            self.switched = True
            self.on_defective()


# --- watcher: RobotRuntime callback ------------------------------------------


def brick_in_jaws(wrist_rgb: np.ndarray) -> bool:
    """A coloured brick in the gap right of the static jaw, close enough to grasp (wrist camera, 640x480).

    Measured on the day's grasps: good closes had a ~19k px brick blob centred at
    x 290-400; misses had it left of the static jaw (x < 200, the jaw lands on it)
    or small under the jaw tip. The claw's shadow is grey, so it never counts.
    """
    hsv = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(np.int16) for i in range(3))
    pink = ((h >= 140) | (h <= 8)) & (s > 70)
    colour = ((((h >= 25) & (h <= 85) & (s > 50) & (v > 70))           # green
               | ((h >= 112) & (h <= 139) & (s > 40) & (v > 90))      # lavender
               | ((h >= 95) & (h <= 125) & (s > 110) & (v > 60)))     # blue
              & ~pink)
    mask = cv2.morphologyEx(colour.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    sx, sy = wrist_rgb.shape[1] / 640, wrist_rgb.shape[0] / 480
    for k in range(1, n):
        cx, cy = cents[k][0] / sx, cents[k][1] / sy
        if stats[k][4] / (sx * sy) > 10000 and 270 <= cx <= 520 and 150 <= cy <= 330:
            return True
    return False


class Watcher:
    """Spots holds and decides when a policy run is over.

    end="home":  stop when the arm has left home and is back there, still (the
                 whole demo ran; with no inspection hold it was a missed grasp).
    end="holds": stop once `holds` inspection holds have been scored, or as
                 soon as a face has failed and the brick has been held at the
                 inspection camera (it is grasped); then the drop policy or the
                 scripted drop takes over.
    """

    # Studio's normalized joint ranges: body joints -100..100, gripper 0..100.
    LOW = np.array([-100, -100, -100, -100, -100, 0], np.float32)
    HIGH = np.full(6, 100, np.float32)
    GRIP_CLOSED = 5.0  # gripper commands at or below this are "closing", left alone by gripper_gain

    def __init__(self, inspector: Inspector, cfg: dict, fps: float):
        self.inspector, self.cfg = inspector, cfg
        self.hold_ticks = max(1, int(float(cfg.get("hold_min_s", 0.3)) * fps))
        self.sample_every = max(1, int(fps / 10))
        self.stop = None  # set to runtime.stop
        self.save_dir: Path | None = None  # set by the cell: runs/<time>/nearspot
        self.gripper_gain = 1.0  # set by the cell from policy.gripper_gain
        self.debug = None  # cell/debugview.py, with --debug
        self.close_gate = False  # set by the cell from policy.close_gate
        self.gated = 0           # actions whose close was held back, this run
        self.source = None
        self._last_debug = 0.0
        self.phase_tag = ""
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
        self.left_home_at: float | None = None
        self.motion: list[tuple] = []  # (t, state, action_sent, loop_s, sleep_s, stale) per tick
        self.speed_ema = 0.0
        self.saved_near = 0
        self.saved_track = 0
        self.last_track_save = 0.0
        self.last_near_save = 0.0
        self.grip_max = 0.0
        self.grasped_at: float | None = None
        self.gated = 0
        self.brick_ready = False
        self.gate_open = False      # the close gate let this attempt's close through
        self._last_gate_check = 0.0

    def on_action_ready(self, *, action, step):
        """Last check before a policy action reaches the arm.

        A NaN/inf action (a broken or untrained export) ends the run instead of
        being sent; everything else is clamped to the calibrated joint range.
        """
        action = np.asarray(action, np.float32)
        if not np.all(np.isfinite(action)):
            raise ValueError(f"policy produced a non-finite action at step {step}: {action}")
        if self.close_gate and self.state is not None and not self.gate_open:
            grip = float(self.state[5])
            if action[5] < grip - 3.0 and grip > 15.0:  # the policy wants to close an open claw
                if self.brick_ready:
                    self.gate_open = True  # brick in the gap: let this grasp happen
                else:
                    action = action.copy()
                    action[5] = grip  # hold the opening; the policy keeps approaching
                    self.gated += 1
        if self.gripper_gain != 1.0 and action[5] > self.GRIP_CLOSED:
            # The policy opens the jaws only to ~20 where the demos opened ~35, so the
            # fingers land on the brick instead of around it. Widen every "open"
            # command; closing commands (near 0) pass through unchanged.
            g = self.GRIP_CLOSED + (action[5] - self.GRIP_CLOSED) * self.gripper_gain
            action = action.copy()
            action[5] = g
        return np.clip(action, self.LOW, self.HIGH)

    def on_tick(self, event):
        state = np.asarray(event.robot_state.joint_positions, np.float32)
        sent = event.action_sent
        self.motion.append((time.monotonic(), state.copy(), None if sent is None else np.asarray(sent, np.float32).copy(),
                            float(event.loop_duration_s), float(event.sleep_time_s), bool(event.stale_obs)))
        now = time.monotonic()
        speed = 0.0
        if self.prev is not None and now > self.prev[0]:
            speed = float(np.max(np.abs(state - self.prev[1]))) / (now - self.prev[0])  # gripper included
        self.prev = (now, state)
        self.state = state
        if self.close_gate and event.camera_frames and "wrist" in event.camera_frames and now - self._last_gate_check > 0.1:
            self._last_gate_check = now
            self.brick_ready = brick_in_jaws(event.camera_frames["wrist"].data)
        if self.debug is not None and now - self._last_debug > 0.1:
            self._last_debug = now
            plan = None
            try:
                plan = self.source.action_queue.peek_remaining() if self.source is not None else None
            except Exception:  # noqa: BLE001
                pass
            self.debug.update(state=state, cmd=sent, plan=plan)
        # Smoothed over ~0.15 s: per-tick speed at 90 Hz is noisy.
        self.speed_ema = 0.85 * self.speed_ema + 0.15 * speed
        if self.done:
            return
        # The policy rarely stops as dead still as the demos (offline extraction uses
        # still_speed 15 on them), so live holds use live_still_speed, and a looser
        # limit right at an inspection spot so a slow pass still gets scored.
        at_spot = self.at_spot(state)
        limit = float(self.cfg.get("live_still_speed", 35.0))
        if at_spot:
            limit = max(limit, float(self.cfg.get("live_spot_speed", 60.0)))
        still = self.speed_ema <= limit
        self.still = self.still + 1 if still else 0
        if at_spot and event.camera_frames:
            self._save_near_spot(now, event.camera_frames)
        elif event.camera_frames:
            self._save_track(now, event.camera_frames)

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
            tol = float(self.cfg["home_tolerance"])
            at_home = self.home is not None and near(state, self.home, tol)
            # "Left home" means really going somewhere: a shake at the start (a few
            # units, back again) once ended the brick after 1-4 s. Back home must be
            # still by the strict demo speed, not the lenient live hold speed.
            if self.home is not None and not near(state, self.home, max(20.0, 2 * tol)):
                self.left_home_at = self.left_home_at or now
            self.left_home = self.left_home_at is not None and now - self.left_home_at >= 5.0
            home_still = self.speed_ema <= float(self.cfg["still_speed"])
            self.home_still = self.home_still + 1 if at_home and home_still else 0
            if self.left_home and self.home_still >= self.hold_ticks:
                self._finish()
        elif self.end == "lifted":
            # Grasp-only policy ("pick up the brick"): done once the gripper has closed
            # well below its widest and, a moment later, the arm is held still (lifted).
            self.grip_max = max(self.grip_max, float(state[5]))
            # Closed for real: opened wide first, then shut (a held brick reads ~2, empty ~1.2).
            # A relative drop fired on the policy just adjusting the opening (50 -> 41).
            if (self.grasped_at is None and self.grip_max > float(self.cfg.get("grasp_open_min", 25.0))
                    and float(state[5]) < float(self.cfg.get("grasp_closed_max", 12.0))):
                self.grasped_at = now
            elif self.grasped_at is not None and float(state[5]) > float(self.cfg.get("grasp_closed_max", 12.0)):
                self.grasped_at = None  # opened again: not holding yet
            if (self.grasped_at is not None and now - self.grasped_at >= float(self.cfg.get("lift_wait_s", 1.0))
                    and self.still >= self.hold_ticks):
                self._finish()
        elif self.end == "holds":
            v = self.inspector.verdict
            # A failed face ends the pick early, but only once the brick is in the
            # gripper: a defect seen by the wrist camera during the hover comes
            # before the grasp, and handing over then would drop an empty gripper.
            if v.holds >= self.holds_wanted or (self.inspector.switched and v.holds >= 1):
                self._finish()

    def at_spot(self, state: np.ndarray) -> bool:
        """Within pose_tolerance of any inspection view's recorded spot pose."""
        tol = float(self.cfg.get("pose_tolerance", 10.0))
        for v in self.cfg["views"].values():
            if not v.get("hover") and at_pose(state, v, tol) is not None:
                return True
        return False

    def _save_near_spot(self, now: float, frames: dict):
        """Keep inspection and wrist frames ~3x/s near a spot (max 60 per run), for review."""
        if self.save_dir is None or self.saved_near >= 60 or now - self.last_near_save < 0.33:
            return
        self.last_near_save = now
        self.saved_near += 1
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for cam in ("inspection", "wrist"):
            if cam in frames:
                cv2.imwrite(str(self.save_dir / f"{self.phase_tag}_{self.saved_near:03d}_{cam}.jpg"),
                            cv2.cvtColor(frames[cam].data, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _save_track(self, now: float, frames: dict):
        """Top and wrist frames once a second away from the spots (max 180), to see grasp attempts afterwards."""
        if self.save_dir is None or self.saved_track >= 180 or now - self.last_track_save < 1.0:
            return
        self.last_track_save = now
        self.saved_track += 1
        d = self.save_dir.parent / "track"
        d.mkdir(parents=True, exist_ok=True)
        g = float(self.state[5]) if self.state is not None else float("nan")
        for cam in ("top", "wrist"):
            if cam in frames:
                cv2.imwrite(str(d / f"{self.phase_tag}_{time.strftime('%H%M%S')}_g{g:02.0f}_{cam}.jpg"),
                            cv2.cvtColor(frames[cam].data, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])

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
    tag_shift: float = 0.0       # px a camera moved whose inspection crops followed the AprilTags
    tag_note: str = ""           # what the AprilTag check found (cell/tags.py)
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

    def __call__(self, *args, **kwargs):  # RTCExecution calls the model directly
        return self._model(*args, **kwargs)

    def close(self):
        pass


def rtc_source(model, pcfg: dict, task: str):
    """PolicySource with Real-Time Chunking (Robotics AI Suite, pi05_with_rtc.md).

    Inference runs on a background thread while the arm plays the queued
    actions, so the loop never waits for the policy. Each new chunk is denoised
    to continue the unexecuted tail of the previous one (prefix guidance), so
    there is no stale-chunk jump like plain async mode had. Needs an export
    with RTC inputs: scripts/export_rtc.py.
    """
    from physicalai.inference.callbacks import RTCLatencyTracker
    from physicalai.runtime import PolicySource, RTCActionQueue, RTCExecution

    rcfg = pcfg.get("rtc") or {}
    # The tracker measures inference latency; RTCExecution turns its worst case
    # into how many queued actions will already be gone when the chunk lands.
    tracker = next((c for c in model.callbacks if isinstance(c, RTCLatencyTracker)), None)
    if tracker is None:
        tracker = RTCLatencyTracker(window_size=int(rcfg.get("latency_window", 30)))
        model.callbacks.append(tracker)
    execution = RTCExecution(
        fps=float(pcfg["fps"]),
        execution_horizon=int(rcfg.get("execution_horizon", 15)),
        max_guidance_weight=float(rcfg.get("max_guidance_weight", 5.0)),
        latency_tracker=tracker,
    )
    return PolicySource(KeepOpen(model), execution, RTCActionQueue(), task=task)


def rtc_export_dir(export_dir: Path) -> Path:
    """The RTC export next to a plain one (exports/openvino -> exports/openvino-rtc)."""
    if _has_rtc_inputs(export_dir):
        return export_dir
    rtc = export_dir.with_name(export_dir.name + "-rtc")
    if (rtc / "manifest.json").exists() and _has_rtc_inputs(rtc):
        return rtc
    raise SystemExit(f"policy.mode: rtc needs an RTC export of {export_dir}; make it with "
                     f"scripts/export_rtc.py {export_dir.parent.parent}")


def _has_rtc_inputs(export_dir: Path) -> bool:
    import openvino as ov

    xml = next((p for p in export_dir.glob("*.xml") if "tokenizer" not in p.name), None)
    if xml is None:
        return False
    names = {n for i in ov.Core().read_model(xml).inputs for n in i.get_names()}
    return "prev_chunk_left_over" in names


class Cell:
    def __init__(self, cfg: dict, arm, cameras: dict, model, anomaly: dict[str, AnomalyModel], poses: dict,
                 log: RunLog, show: bool = False, drop_model=None, guard: tags_mod.TagGuard | None = None):
        from physicalai.runtime import AsyncExecution, PolicySource, RobotRuntime, SyncExecution

        self.cfg, self.arm, self.cameras, self.poses, self.log = cfg, arm, cameras, poses, log
        self.guard = guard
        self.pcfg, self.icfg = cfg["policy"], cfg["inspection"]
        self.views = self.icfg["views"]  # this brick's views: crop boxes follow the tags when a camera moved
        self.show = show
        self.mode = "handover" if drop_model is not None else self.pcfg["drop"]
        needed = ["home"] + (["good_bowl", "defective_bowl"] if self.mode in ("scripted", "grasp") else [])
        missing = [p for p in needed if p not in poses]
        if missing:
            raise SystemExit(f"poses missing from {cfg['poses']}: {missing}; record them with ./biddy poses record")
        self.insp_views, self.hover_view = split_views(self.icfg["views"])
        if not any(v in anomaly for v in self.insp_views):
            raise SystemExit(f"no anomaly model for any inspection view {self.insp_views}; train with ./biddy anomaly")

        def source(m):
            mode = self.pcfg.get("mode", "sync")
            if mode == "rtc":
                return rtc_source(m, self.pcfg, self.pick_task)
            execution = AsyncExecution() if mode == "async" else SyncExecution()
            return PolicySource(KeepOpen(m), execution, task=self.pick_task)

        self.tasks = self.pcfg["tasks"]
        self.pick_task = self.tasks.get("pick") or self.tasks["good"]
        # --collect good|defective: a known brick; never switch bowls, keep every hold's crops.
        self.collect = cfg.get("collect")
        if self.mode == "grasp":  # grasp-only policy; the script inspects and drops
            self.pick_task = self.tasks.get("grasp") or "pick up the brick"
        elif self.collect:
            self.pick_task = self.tasks[self.collect]
        self.models = [model] + ([drop_model] if drop_model is not None else [])
        pick_source = source(model)
        self.source = SwitchSource({"pick": pick_source,
                                    "drop": source(drop_model) if drop_model is not None else pick_source})
        self.inspector = Inspector(anomaly, self.icfg["views"], self.icfg, self._on_defective)
        if self.collect:
            self.inspector.save_dir = Path("data/live") / self.collect
        from physicalai.runtime import LinearInterpolator

        # The SO-101 ignores goal_time and jumps to each target, so 30 Hz policy
        # actions make a staircase (jitter). Split every action into `interpolate`
        # linearly interpolated commands (3 -> 90 Hz), as Studio's own robot loop does.
        self.interpolate = int(self.pcfg.get("interpolate", 3))
        from physicalai.runtime import LowPassFilterCallback

        tick_hz = float(self.pcfg["fps"]) * self.interpolate
        self.watcher = Watcher(self.inspector, self.icfg, tick_hz)
        self.watcher.save_dir = log.dir / "nearspot"
        self.watcher.gripper_gain = float(self.pcfg.get("gripper_gain", 1.0))
        # Wrist-camera close gate: the claw only closes with a coloured brick in the jaw gap
        # (the policy otherwise closes on the claw's own shadow on the white card).
        self.watcher.close_gate = bool(self.pcfg.get("close_gate", False))
        # The fine-tuned policy's actions are ~7x jerkier than the demos; an EMA on
        # the 30 Hz actions (reset at every run) smooths them before the safety
        # check. Lower alpha = smoother but laggier; 1 = off.
        callbacks = [self.watcher]
        alpha = float(self.pcfg.get("smoothing", 0.35))
        if alpha < 1.0:
            callbacks.insert(0, LowPassFilterCallback(alpha=alpha))
        self.runtime = RobotRuntime(robot=arm, action_source=self.source, fps=float(self.pcfg["fps"]),
                                    cameras=cameras, callbacks=callbacks,
                                    interpolator=LinearInterpolator(multiplier=self.interpolate))
        self.watcher.stop = self.runtime.stop
        self.watcher.source = self.source
        self.debug = None  # cell/debugview.py, with --debug
        self.insp_cam = cameras[self.icfg["views"][self.insp_views[0]]["camera"]]
        self.stop_event = threading.Event()

    def _on_defective(self):
        """Inspector thread: a face failed. In task_switch mode, steer to the defective bowl now."""
        if self.mode == "task_switch" and not self.collect:
            self.source.current.set_task(self.tasks["defective"])

    def _dbg(self, phase: str, detail: str = ""):
        if self.debug is not None:
            self.debug.set_phase(phase, detail)

    def _policy(self, phase: str, task: str, timeout: float, end: str, holds: int = 0) -> tuple[bool, float]:
        self._dbg(f"policy {phase}", f'"{task}"')
        self.source.use(phase)
        self.source.current.set_task(task)
        self.watcher.begin(end, self.poses["home"], holds)
        self.watcher.phase_tag = f"b{len(self.log.rows) + 1:03d}_{phase}"
        t0 = time.monotonic()
        self.runtime.run(duration_s=timeout, stop_event=self.stop_event)
        self._save_motion(phase)
        # run() swallows Ctrl+C and returns; don't let that look like a timeout.
        if self.runtime.last_run_reason in ("interrupted", "error") or self.stop_event.is_set():
            self.stop_event.set()
            raise StopCell(f"policy run ended: {self.runtime.last_run_reason}")
        return self.watcher.done, time.monotonic() - t0

    def _save_motion(self, phase: str):
        """runs/<time>/motion_<brick>_<phase>.csv: what the arm did and was told, per tick,
        plus a one-line timing check (late ticks mean the laptop is too busy for smooth control)."""
        rows = self.watcher.motion
        if not rows:
            return
        brick = len(self.log.rows) + 1
        path = self.log.dir / f"motion_{brick:04d}_{phase}.csv"
        n = 2
        while path.exists():  # retries of the same brick keep their own file
            path = self.log.dir / f"motion_{brick:04d}_{phase}_{n}.csv"
            n += 1
        names = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
        t0 = rows[0][0]
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", *[f"state_{n}" for n in names], *[f"cmd_{n}" for n in names], "loop_s", "sleep_s", "stale"])
            for t, st, cmd, loop_s, sleep_s, stale in rows:
                cmd_vals = [""] * 6 if cmd is None else [round(float(x), 3) for x in cmd]
                w.writerow([round(t - t0, 4), *[round(float(x), 3) for x in st], *cmd_vals,
                            round(loop_s, 5), round(sleep_s, 5), int(stale)])
        late = sum(1 for r in rows if r[4] < 0)
        grip_cmd = [r[2][5] for r in rows if r[2] is not None]
        grip = [r[1][5] for r in rows]
        print(f"  motion: {len(rows)} ticks in {rows[-1][0] - t0:.1f}s, {late} late ({100 * late / len(rows):.0f}%), "
              f"gripper commanded {min(grip_cmd, default=float('nan')):.1f}-{max(grip_cmd, default=float('nan')):.1f}, "
              f"measured {min(grip):.1f}-{max(grip):.1f} -> {path.name}"
              + (f"; close gate held {self.watcher.gated} closes" if self.watcher.close_gate else ""))

    def _move(self, pose: np.ndarray, gripper: float | None = None, tolerance: float = 3.0) -> float:
        """Scripted move; raises StopCell if it is stopped or does not arrive."""
        t0 = time.monotonic()
        ok = poses_mod.move_to(self.arm, pose, speed=float(self.cfg["scripted"]["speed"]), gripper=gripper,
                               stop=self.stop_event, tolerance=tolerance)
        if not ok:
            raise StopCell("stopped" if self.stop_event.is_set() else
                           f"arm did not reach {np.round(pose, 1).tolist()}; check for a collision or a stuck joint")
        return time.monotonic() - t0

    def _preview(self, rgb: np.ndarray, verdict: str, score: float):
        if not self.show:
            return
        img = overlay(rgb, Score(score, verdict == "defective", 0.0))
        try:
            cv2.imshow("worst face", img)
            cv2.waitKey(1)
        except cv2.error:
            # Studio's venv has a headless OpenCV (no windows): keep the latest preview as a file instead.
            path = self.log.dir / "preview.png"
            cv2.imwrite(str(path), img)
            if not getattr(self, "_preview_note", False):
                print(f"(no display support in OpenCV; --show writes {path} instead)")
                self._preview_note = True

    def _check_tags(self, c: Cycle) -> float:
        """Arm at home: compare the AprilTags with their references (cell/tags.py).

        Sets this brick's crop boxes. With tags.on_moved: pause, waits until
        everything that moved is back. Returns the seconds spent waiting.
        """
        self.views = self.icfg["views"]
        if self.guard is None:
            return 0.0
        waiting_since = None
        while True:
            report = self.guard.check(self.cameras)
            if not report.problems or self.guard.on_moved != "pause":
                break
            if waiting_since is None:
                waiting_since = time.monotonic()
                print(f"brick {c.cycle}: WAITING: {'; '.join(report.problems)}. Put it back (Ctrl+C stops).")
            if self.stop_event.wait(1.0):
                raise StopCell("stopped while waiting for a moved tag")
        waited = time.monotonic() - waiting_since if waiting_since is not None else 0.0
        self.views, c.tag_shift = tags_mod.follow(self.icfg["views"], report.matrices), report.shift
        notes = report.problems + report.notes + ([f"waited {waited:.0f}s for a moved tag"] if waited else [])
        c.tag_note = "; ".join(notes)
        return waited

    def _inspection_stops(self) -> list[tuple[str, int, np.ndarray]]:
        """(view, index, 5-joint pose): the first two recorded poses of each inspection view,
        skipping the long-brick ones (wrist rolled ~50)."""
        stops = [(v, i, np.asarray(p, np.float32)) for v in self.insp_views
                 for i, p in enumerate(self.icfg["views"][v].get("poses") or []) if abs(float(p[4])) < 20]
        return [s for n, s in enumerate(stops) if sum(1 for t in stops[:n] if t[0] == s[0]) < 2]

    def _scripted_inspection(self, c: Cycle, empty: dict) -> bool:
        """Grasp mode: show the held brick at fixed poses and score it there.

        Every crop has the same framing (the anomaly models are trained on these
        poses), unlike holds the policy happens to pause in. Fills the verdict
        through the inspector; returns False if no pose showed a brick.
        """
        views = self.views
        n_frames = int(self.icfg.get("frames", 3))
        settle = float(self.icfg.get("scripted_settle_s", 0.6))
        insp = self.inspector
        # Holds the watcher saw during the grasp don't count: only the fixed poses do.
        insp.reset(self.poses["home"], insp.backgrounds, views, empty)
        t0 = time.monotonic()
        found = self._show_at_stops(empty, n_frames, settle, base=0)
        # Regrip only a brick that passed so far: a failed face already decides "defective".
        if self.icfg.get("scripted_regrip") and found and (self.collect or not insp.verdict.defective):
            if self._regrip():
                found += self._show_at_stops(empty, n_frames, settle, base=100, tag="after regrip ")
        c.episode_s += time.monotonic() - t0
        return found > 0

    def _regrip(self) -> bool:
        """Set the brick down at the regrip spot, roll the wrist, grasp it again.

        The script knows where the brick is (it is holding it), so this needs no
        vision: down at poses.regrip, open, up to regrip_above, roll by
        inspection.regrip_roll (normalized units; ~50 = 90 degrees), down, close,
        up. The faces the jaws covered are then shown at the same poses.
        """
        self._dbg("regrip", "set down, roll, grasp again")
        if "regrip" not in self.poses or "regrip_above" not in self.poses:
            print("  regrip: poses regrip/regrip_above not recorded; skipping")
            return False
        open_g = float(self.cfg["scripted"]["gripper_open"])
        roll = float(self.icfg.get("regrip_roll", 50.0))
        above, down = self.poses["regrip_above"].copy(), self.poses["regrip"].copy()
        above_r, down_r = above.copy(), down.copy()
        above_r[4] = float(np.clip(above_r[4] + roll, -100, 100))
        down_r[4] = float(np.clip(down_r[4] + roll, -100, 100))
        self._move(above, gripper=0.0, tolerance=8.0)
        self._move(down, gripper=0.0, tolerance=8.0)
        self._move(down, gripper=open_g, tolerance=8.0)
        self._move(above, gripper=open_g, tolerance=8.0)
        self._move(above_r, gripper=open_g, tolerance=8.0)
        self._move(down_r, gripper=open_g, tolerance=8.0)
        self._move(down_r, gripper=0.0, tolerance=8.0)
        time.sleep(0.3)
        self._move(above_r, gripper=0.0, tolerance=8.0)
        print(f"  regrip: rolled {roll:+.0f} and grasped again")
        return True

    def _show_at_stops(self, empty: dict, n_frames: int, settle: float, base: int, tag: str = "") -> int:
        """Show the held brick at every inspection stop; returns how many showed a brick."""
        views, insp = self.views, self.inspector
        found = 0
        for k, (view, i, p) in enumerate(self._inspection_stops()):
            if view not in insp.models:
                continue
            self._dbg("inspect", f"{tag}{view} pose {i}")
            self._move(np.concatenate([p[:5], [0.0]]).astype(np.float32), gripper=0.0)
            time.sleep(settle)
            cam = views[view]["camera"]
            samples = []
            for _ in range(n_frames):
                rgb = score_crop(self.cameras[cam].read().data, views[view], empty.get(cam))
                if rgb is None:
                    continue
                r = insp.models[view](rgb)
                insp.infer_ms.append(r.latency_ms)
                samples.append((r.score, rgb))
                time.sleep(0.1)
            if not samples:
                print(f"  {tag}{view} pose {i}: no brick found")
                continue
            found += 1
            with insp.lock:
                hold_id = 1000 + base + k
                insp.kinds[hold_id] = view
                insp.samples[hold_id] = samples
                insp._close(hold_id)  # median into the verdict; saves crops in --collect
            print(f"  {tag}{view} pose {i}: " + " ".join(f"{s:.3f}" for s, _ in samples))
        return found

    # one brick

    def cycle(self, i: int) -> Cycle:
        c = Cycle(i, started=datetime.now().isoformat(timespec="seconds"))
        t_start = time.monotonic()
        t_start += self._check_tags(c)  # time spent waiting for a person is not cycle time
        if hasattr(self.insp_cam, "next_brick"):  # mock: put the next brick on the table
            c.truth = self.insp_cam.next_brick()
        views = self.views
        timeout = float(self.pcfg["episode_timeout_s"])
        holds_wanted = int(self.icfg.get("holds_per_brick", 4))

        for attempt in range(int(self.icfg.get("max_regrasps", 0)) + 1):
            c.regrasps = attempt
            # Arm at home: the inspection view is empty; that is the presence reference.
            frames = {cam: self.cameras[cam].read().data for cam in {views[v]["camera"] for v in self.insp_views}}
            if self.debug is not None and "top" in self.cameras:
                self.debug.set_empty_top(None)  # the bricks are on the table now: diff against the first frame of the run
            self.inspector.reset(self.poses["home"], {v: view_crop(frames[views[v]["camera"]], views[v])
                                                      for v in self.insp_views}, views, frames)
            if self.mode == "grasp":
                done, c.episode_s = self._policy("pick", self.pick_task, timeout, "lifted")
                if not done:
                    c.outcome, c.note = "no_brick", f"no grasp within {timeout:.0f}s"
                    c.home_s = self._move(self.poses["home"], gripper=float(self.cfg["scripted"]["gripper_open"]))
                    continue
                c.note = ""  # an earlier missed attempt's note
                found = self._scripted_inspection(c, frames)
                if not found:
                    c.outcome, c.note = "no_brick", "no brick in any inspection crop"
                    c.home_s = self._move(self.poses["home"], gripper=float(self.cfg["scripted"]["gripper_open"]))
                    continue
                break
            if self.mode == "task_switch":
                done, c.episode_s = self._policy("pick", self.pick_task, timeout, "home")
            else:
                done, c.episode_s = self._policy("pick", self.pick_task, timeout, "holds", holds_wanted)
            self.inspector.sync()
            v = self.inspector.verdict
            # No inspection hold is a missed grasp, unless the wrist camera already
            # found the defect: demos may go straight to the defective bowl then.
            if v.holds == 0 and not v.defective:
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

        if self.mode == "grasp":
            t0 = time.monotonic()
            label = self.collect or c.verdict  # --collect: the known bowl
            self._dbg("drop", f"{label} bowl (verdict {c.verdict}, worst {c.worst_view} {c.score:.3f})")
            bowl = self.poses[f"{label}_bowl"]
            # Stretched out over a bowl the arm sags several units: arriving roughly is enough.
            self._move(bowl, gripper=0.0, tolerance=8.0)  # keep squeezing the brick
            self._move(bowl, gripper=float(self.cfg["scripted"]["gripper_open"]), tolerance=8.0)
            c.drop_s = time.monotonic() - t0
            c.home_s = self._move(self.poses["home"])
            c.outcome = "sorted"
        elif self.mode == "task_switch":
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
                      + (f" (truth {c.truth})" if c.truth else "") + (f"  {c.note}" if c.note else "")
                      + (f"  tags: {c.tag_note}" if c.tag_note else ""))


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
        if cfg["policy"].get("mode") == "rtc":
            export_dir = rtc_export_dir(export_dir)
        check_cameras(export_dir, cfg["studio"]["cameras"])
        return load_policy(export_dir, cfg, role)

    model = load(cfg["policy"]["model"], "pick")
    drop = cfg["policy"].get("drop_model")
    drop_model = load(drop, "drop") if drop else None
    return build_arm(cfg), build_cameras(cfg), model, drop_model


def build_mock(cfg: dict, poses: dict, miss_every: int):
    from cell.hardware import MOCK_POSES, MockArm, MockCamera, MockPolicy, MockStation

    arm = MockArm(start=poses["home"])
    views = cfg["inspection"]["views"]
    # The mock arm follows its own route, so its spots replace the station's
    # recorded spot poses and crop boxes (mock only; station.yaml is untouched).
    mock_views = {
        "side": {"camera": "inspection", "roi": [240, 160, 160, 160],
                 "poses": [MOCK_POSES[p][:5] for p in ("inspect_back", "inspect_side1", "inspect_side2")]},
        "bottom": {"camera": "inspection", "roi": [450, 310, 160, 160], "poses": [MOCK_POSES["inspect_bottom"][:5]]},
        "wrist": {"camera": "wrist", "roi": [260, 280, 120, 120], "hover": True},
    }
    for name, v in views.items():
        if name in mock_views:
            v.update(mock_views[name])
            v.pop("crop", None)  # the mock draws red bricks, which the brick finder treats as gripper
    cams = {name: MockCamera(name) for name in cfg["studio"]["cameras"]}
    station = MockStation(arm, poses, views, miss_every=miss_every)
    for view in views:
        cams[views[view]["camera"]] = station.camera(view)
    return arm, cams, MockPolicy(poses, cfg["policy"]["tasks"], fps=float(cfg["policy"]["fps"])), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--cycles", type=int, help="stop after this many bricks (default: until Ctrl+C)")
    ap.add_argument("--drop", choices=["task_switch", "scripted", "grasp"], help="override policy.drop")
    ap.add_argument("--policy-device", help="override policy.device")
    ap.add_argument("--anomaly-device", help="override inspection.device")
    ap.add_argument("--policy-mode", choices=["sync", "async", "rtc"], help="override policy.mode")
    ap.add_argument("--show", action="store_true", help="preview window with each brick's worst face")
    ap.add_argument("--mock", action="store_true", help="no hardware: mock arm, cameras and policy")
    ap.add_argument("--mock-miss-every", type=int, default=0, help="mock: every n-th pick grabs nothing")
    ap.add_argument("--mock-policy", help="mock: drive the mock arm with this exported policy instead of the scripted one")
    ap.add_argument("--collect", choices=["good", "defective"],
                    help="known brick: stay on that bowl's task and save every scored crop to data/live/<label>/<view>/")
    ap.add_argument("--debug", action="store_true",
                    help="debug frames even if switched off on the debug page (see ./biddy debug-server)")
    ap.add_argument("--no-tags", action="store_true", help="skip the AprilTag check and the crops that follow the tags (station.yaml tags)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.drop:
        cfg["policy"]["drop"] = args.drop
    if args.policy_device:
        cfg["policy"]["device"] = args.policy_device
    if args.policy_mode:
        cfg["policy"]["mode"] = args.policy_mode
    if args.anomaly_device:
        cfg["inspection"]["device"] = args.anomaly_device
    cfg["collect"] = args.collect

    poses = {k: np.asarray(v, np.float32) for k, v in poses_mod.load(cfg["poses"]).items()}
    if args.mock:
        from cell.hardware import MOCK_POSES

        poses = {k: np.asarray(v, np.float32) for k, v in MOCK_POSES.items()}  # the mock policy's own route
        arm, cameras, model, drop_model = build_mock(cfg, poses, args.mock_miss_every)
        if args.mock_policy:  # real exported policy (InferenceModel) driving the mock arm
            mock_dir = studio.resolve_policy_dir(args.mock_policy, cfg["studio"]["db"])
            if cfg["policy"].get("mode") == "rtc":
                mock_dir = rtc_export_dir(mock_dir)
            check_cameras(mock_dir, cfg["studio"]["cameras"])
            model = load_policy(mock_dir, cfg)
    else:
        arm, cameras, model, drop_model = build_real(cfg)
    anomaly = load_anomaly(cfg)
    # The mock cameras show no tags, so the check is for the real station only.
    guard = None if args.mock or args.no_tags else tags_mod.TagGuard.from_config(cfg.get("tags"))
    if guard is not None:
        print(f"tags: checking {', '.join(guard.cameras)} before every brick ({guard.on_moved} when one moved)"
              + (f"; {', '.join(sorted(guard.correct))} crops follow the tags" if guard.correct else ""))

    log = RunLog(cfg["runs_dir"], cfg)
    cell = Cell(cfg, arm, cameras, model, anomaly, poses, log, show=args.show, drop_model=drop_model, guard=guard)
    # Debug frames go to runs/debug_latest.jpg for the always-on page (scripts/debug_server.py,
    # ./biddy debug-server), unless it was switched off there (runs/.debug_off). --debug forces it.
    runs = Path(cfg["runs_dir"])
    if args.debug or not (runs / ".debug_off").exists():
        from cell.debugview import DebugView

        cell.debug = cell.watcher.debug = DebugView(cameras, log.dir, latest=runs / "debug_latest.jpg")
    try:
        cell.run(args.cycles)
    except KeyboardInterrupt:
        cell.stop_event.set()
        print("\nstopped")
    finally:
        if cell.debug is not None:
            cell.debug.close()
        for m in cell.models:
            close = getattr(m, "close", None)
            if close is not None:
                close()
        print(log.summary())


if __name__ == "__main__":
    main()
