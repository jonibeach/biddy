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
    return SO101(port=studio.find_port(rec), calibration=SO101Calibration.from_dict(rec.calibration),
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
        # Shuffled (fixed seed) so good and defective bricks are mixed like on the table.
        np.random.default_rng(0).shuffle(paths)
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


# The mock policy's route, following docs/data_collection.md (normalized units).
MOCK_POSES = {
    "home": [0, -90, 90, 60, 0, 60],
    "hover": [10, -40, 60, 40, 0, 60],
    "inspect_back": [30, -20, 30, 10, 0, 5],
    "inspect_bottom": [30, -20, 30, 45, 0, 5],
    "regrasp": [-10, -30, 50, 40, 0, 5],
    "inspect_side1": [30, -20, 30, 10, -60, 5],
    "inspect_side2": [30, -20, 30, 10, 60, 5],
    "good_bowl": [-50, -10, 20, 30, 0, 5],
    "defective_bowl": [60, -10, 20, 30, 0, 5],
}
FACES = {"inspect_back": "back", "inspect_bottom": "bottom", "inspect_side1": "side1", "inspect_side2": "side2"}


class MockStation:
    """The table, the brick and what each camera sees, driven by the mock arm's pose.

    Each brick is good or has a mark on one face (back, bottom, side1, side2
    or top). The inspection camera shows the face for the inspection pose the
    arm is at; the wrist camera shows the top face during the hover. Images are
    drawn in the same style as the synthetic test dataset, inside each view's
    ROI, so models trained on that dataset can judge them. `miss_every` makes
    every n-th brick's first grasp come up empty.
    """

    COLORS = {"back": (200, 40, 40), "bottom": (190, 50, 50), "side1": (205, 35, 45),
              "side2": (195, 45, 35), "top": (200, 40, 40)}

    def __init__(self, arm, poses: dict, views: dict, miss_every: int = 0, seed: int = 0):
        self.arm, self.views, self.miss_every = arm, views, miss_every
        self.poses = {k: np.asarray(v, np.float32) for k, v in poses.items()}
        self.rng = np.random.default_rng(seed)
        self.bricks = 0
        self.defect: str | None = None
        self.missed = False

    def next_brick(self) -> str:
        self.bricks += 1
        self.defect = str(self.rng.choice(["back", "bottom", "side1", "side2", "top"])) if self.rng.random() < 0.5 else None
        self.missed = bool(self.miss_every) and self.bricks % self.miss_every == 0
        return "defective" if self.defect else "good"

    def regrasp(self):
        self.missed = False

    def _at(self, name: str) -> bool:
        return float(np.max(np.abs(self.arm._pos[:5] - self.poses[name][:5]))) <= 3.0

    def _noise(self, img: np.ndarray) -> np.ndarray:
        return cv2.add(img, self.rng.integers(0, 12, img.shape, dtype=np.uint8))

    def image(self, view: str) -> np.ndarray:
        if view == "wrist":
            img = np.full((480, 640, 3), 110, np.uint8)
            if self._at("hover") and self.bricks:
                cv2.rectangle(img, (280, 300), (360, 380), self.COLORS["top"], -1)
                if self.defect == "top":
                    cv2.circle(img, (320, 340), 12, (0, 0, 0), -1)
            return self._noise(img)
        img = np.full((480, 640, 3), 60, np.uint8)
        face = next((f for pose, f in FACES.items() if self._at(pose)), None)
        if face and self.bricks and not self.missed:
            # Like the station: the bottom is shown at the bottom right, the other faces in the centre.
            x, y = (470, 330) if face == "bottom" else (260, 180)
            cv2.rectangle(img, (x, y), (x + 120, y + 120), self.COLORS[face], -1)
            if self.defect == face:
                cv2.line(img, (x + 10, y + 10), (x + 110, y + 110), (0, 0, 0), 5)
        return self._noise(img)

    def camera(self, view: str) -> "MockViewCamera":
        return MockViewCamera(self, view)


class MockViewCamera(MockCamera):
    def __init__(self, station: MockStation, view: str):
        super().__init__(view)
        self.station, self.view = station, view

    def read_latest(self) -> MockFrame:
        self._seq += 1
        return MockFrame(self.station.image(self.view), time.monotonic(), self._seq)

    def next_brick(self) -> str:
        return self.station.next_brick()

    def regrasp(self):
        self.station.regrasp()


class MockPolicy:
    """Stands in for InferenceModel: plays one demo of the recording protocol.

    home -> hover (hold) -> grasp -> back (hold) -> bottom (hold) -> regrasp
    -> side1 (hold) -> side2 (hold) -> the bowl named by the task -> open ->
    home. Like a VLA it reads the task from each observation, so a task switch
    during the holds changes the bowl. reset() (each new run) starts over.
    """

    HOLD_S = {"hover": 0.6, "inspect_back": 1.0, "inspect_bottom": 1.0, "inspect_side1": 1.0, "inspect_side2": 1.0,
              "home": 0.5}

    def __init__(self, poses: dict, tasks: dict[str, str], chunk: int = 10, step: float = 4.0, fps: float = 30.0):
        self.poses = {k: np.asarray(v, np.float32) for k, v in poses.items()}
        self.tasks, self.chunk, self.step, self.fps = tasks, chunk, step, fps
        self.reset()

    def reset(self):
        self.i = 0
        self.held = 0

    def _route(self, task: str | None) -> list[tuple[str, float]]:
        bowl = "defective_bowl" if task == self.tasks["defective"] else "good_bowl"
        return [("hover", 60), ("hover", 5), ("inspect_back", 5), ("inspect_bottom", 5), ("regrasp", 5),
                ("regrasp", 60), ("regrasp", 5), ("inspect_side1", 5), ("inspect_side2", 5), (bowl, 5),
                (bowl, 60), ("home", 60)]

    def _target(self, pos: np.ndarray, task: str | None) -> np.ndarray:
        route = self._route(task)
        name, grip = route[min(self.i, len(route) - 1)]
        target = self.poses[name].copy()
        target[5] = grip
        if float(np.max(np.abs(pos - target))) <= 1.0 and self.i < len(route) - 1:
            hold = int(self.HOLD_S.get(name, 0.0) * self.fps) if route[self.i + 1][0] != name else 0
            if self.held >= hold:
                self.i, self.held = self.i + 1, 0
            else:
                self.held += 1
        return target

    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["state"], np.float32).reshape(-1)[:6].copy()
        task = (obs.get("task") or [None])[0]
        actions = []
        for _ in range(self.chunk):
            pos = pos + np.clip(self._target(pos, task) - pos, -self.step, self.step)
            actions.append(pos.copy())
        return np.stack(actions)
