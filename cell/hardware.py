"""Build the follower arm and cameras, real or mock.

Real hardware is built from Studio's records (cell/studio.py), the same way
Studio builds it for recording. Studio must not be holding the devices: stop
its robot session (close the record/run page) before starting the cell.

The mocks implement the same small surface the cell uses, so the whole loop
can run on a laptop with nothing plugged in (`python -m cell.run --mock`).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cell import studio

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def build_arm(cfg: dict):
    from physicalai_so101_plugin import SO101, SO101Calibration

    rec = studio.robot(cfg["studio"]["follower"], cfg["studio"]["db"])
    if rec.unit != "normalized":
        raise ValueError(f"{rec.name} uses unit {rec.unit!r}; the cell's tolerances assume 'normalized'")
    return SO101(port=studio.find_port(rec), calibration=SO101Calibration(joints=rec.calibration),
                 role="follower", unit=rec.unit)


def build_cameras(cfg: dict) -> dict:
    from physicalai.capture import RealSenseCamera, UVCCamera

    out = {}
    for name, rec in studio.cameras(cfg["studio"]["cameras"], cfg["studio"]["db"]).items():
        if rec.driver == "realsense":
            out[name] = RealSenseCamera(serial_number=str(rec.fingerprint["serial"]),
                                        width=rec.width, height=rec.height, fps=rec.fps)
        elif rec.driver == "usb_camera":
            out[name] = UVCCamera(device=rec.fingerprint, width=rec.width, height=rec.height, fps=rec.fps)
        else:
            raise ValueError(f"camera {name}: driver {rec.driver!r} not supported by the cell")
    return out


# --- mocks -------------------------------------------------------------------


@dataclass
class MockObservation:
    joint_positions: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)
    sensor_data: None = None
    images: None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


class MockArm:
    """Joints move toward the last command at a capped speed, like a slow servo."""

    joint_names = JOINTS
    unit = "normalized"

    def __init__(self, start: np.ndarray | None = None, speed: float = 150.0):
        self._pos = np.zeros(6, np.float32) if start is None else np.asarray(start, np.float32).copy()
        self._target = self._pos.copy()
        self._speed = speed  # units per second
        self._t = time.monotonic()
        self._connected = False
        self.max_speed = speed

    def _advance(self):
        now = time.monotonic()
        step = self._speed * (now - self._t)
        self._t = now
        delta = self._target - self._pos
        self._pos += np.clip(delta, -step, step).astype(np.float32)

    def connect(self):
        self._connected = True
        self._t = time.monotonic()

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_ids(self) -> tuple[str, ...]:
        return ("mock:arm",)

    def get_observation(self) -> MockObservation:
        self._advance()
        return MockObservation(self._pos.copy())

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        self._advance()
        self._target = np.clip(np.asarray(action, np.float32), -100, 100)

    def set_torque(self, *, enabled: bool) -> None:
        pass


@dataclass(frozen=True)
class MockFrame:
    data: np.ndarray
    timestamp: float
    sequence: int


class MockCamera:
    """Serves RGB frames from a folder of images in a loop, or a plain test pattern.

    `advance()` moves to the next image; the cell calls it once per brick so
    the inspection camera shows a new (good or defective) brick each cycle.
    """

    def __init__(self, name: str, folder: str | Path | None = None, width: int = 640, height: int = 480):
        self.name = name
        self.width, self.height = width, height
        paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in {".png", ".jpg"}) if folder else []
        self.paths = paths
        self._cycle = itertools.cycle(paths) if paths else None
        self._seq = 0
        self.current_path: Path | None = None
        self._img = self._pattern()
        self.advance()

    def _pattern(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 90, np.uint8)
        cv2.putText(img, self.name, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return img

    def advance(self):
        if self._cycle is None:
            return
        self.current_path = next(self._cycle)
        bgr = cv2.imread(str(self.current_path))
        self._img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def connect(self, timeout: float = 5.0):
        pass

    def disconnect(self):
        pass

    def read(self, timeout: float = 2.0) -> MockFrame:
        time.sleep(1 / 30)
        return self.read_latest()

    def read_latest(self) -> MockFrame:
        self._seq += 1
        return MockFrame(self._img, time.monotonic(), self._seq)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()


class MockPolicy:
    """Stands in for InferenceModel: heads for a target pose in small steps.

    The target is the inspect pose until `set_task` switches to a bowl task,
    then the matching bowl pose, so task_switch mode can be exercised.
    """

    def __init__(self, poses: dict[str, np.ndarray], tasks: dict[str, str], chunk: int = 10, step: float = 4.0):
        self.poses = poses
        self.tasks = tasks
        self.chunk = chunk
        self.step = step
        self.task = tasks["good"]
        self.inspected = False

    def target(self) -> np.ndarray:
        if not self.inspected:
            return self.poses["inspect"]
        bowl = "defective" if self.task == self.tasks["defective"] else "good"
        return self.poses[f"{bowl}_bowl"]

    def predict_action_chunk(self, state: np.ndarray) -> np.ndarray:
        actions, pos = [], np.asarray(state, np.float32).copy()
        for _ in range(self.chunk):
            pos = pos + np.clip(self.target() - pos, -self.step, self.step)
            actions.append(pos.copy())
        return np.stack(actions)
