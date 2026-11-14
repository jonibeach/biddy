"""Read a Studio-recorded LeRobot v3 dataset: per-episode states, tasks and frames.

States and tasks come straight from the parquet files (fast, no video decode);
frames are decoded only for the indices asked for. Runs in the Studio venv,
which has a LeRobot that matches the recorder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cell import studio


@dataclass
class Episode:
    index: int
    task: str
    fps: float
    state: np.ndarray   # (T, 6) observation.state
    action: np.ndarray  # (T, 6)
    global_index: np.ndarray  # (T,) dataset-wide frame index, for frame()

    @property
    def seconds(self) -> float:
        return len(self.state) / self.fps


def resolve(spec: str, db=studio.DEFAULT_DB) -> Path:
    """'latest' -> newest Studio dataset; a name -> that Studio dataset; else a path."""
    path = Path(spec).expanduser()
    if (path / "meta" / "info.json").exists():
        return path
    import sqlite3

    with sqlite3.connect(f"file:{Path(db).expanduser()}?mode=ro", uri=True) as c:
        if spec == "latest":
            rows = c.execute("select name, path from datasets order by created_at desc").fetchall()
        else:
            rows = c.execute("select name, path from datasets where name = ?", (spec,)).fetchall()
    for _, p in rows:
        if (Path(p) / "meta" / "info.json").exists():
            return Path(p)
    if rows:
        raise FileNotFoundError(
            f"Studio dataset {rows[0][0]!r} has no files at {rows[0][1]} yet. Studio writes them when the "
            "recording session ends: stop recording in Studio first."
        )
    raise FileNotFoundError(f"{spec!r} is neither a dataset folder nor a Studio dataset name")


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(self.info["fps"])
        self.camera_keys = [k for k, v in self.info["features"].items() if v.get("dtype") in ("video", "image")]
        files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no data parquet files under {self.root / 'data'}")
        self.frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        tasks = pd.read_parquet(self.root / "meta" / "tasks.parquet")
        self.task_names = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
        self._lerobot = None

    def episodes(self) -> list[Episode]:
        out = []
        for ep, df in self.frames.sort_values("index").groupby("episode_index"):
            task_ids = df["task_index"].unique()
            task = " | ".join(self.task_names.get(int(t), f"task {t}") for t in task_ids)
            out.append(Episode(int(ep), task, self.fps,
                               np.stack(df["observation.state"].to_numpy()).astype(np.float32),
                               np.stack(df["action"].to_numpy()).astype(np.float32),
                               df["index"].to_numpy()))
        return out

    def frame(self, global_index: int, camera: str) -> np.ndarray:
        """RGB uint8 (H, W, 3) image of one camera at one dataset-wide frame index."""
        if self._lerobot is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._lerobot = LeRobotDataset("local/dataset", root=self.root, return_uint8=True)
        key = camera if camera.startswith("observation.images.") else f"observation.images.{camera}"
        img = self._lerobot[int(global_index)][key]
        img = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
        if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = img.transpose(1, 2, 0)  # CHW -> HWC
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(img)


def joint_speed(state: np.ndarray, fps: float) -> np.ndarray:
    """(T,) largest body-joint speed in units/s, lightly smoothed."""
    body = state[:, :5]
    speed = np.zeros(len(state), np.float32)
    if len(state) > 1:
        speed[1:] = np.abs(np.diff(body, axis=0)).max(axis=1) * fps
    n = min(5, len(speed))  # a longer kernel than the input changes the output length
    if n <= 1:
        return speed
    return np.convolve(speed, np.ones(n) / n, mode="same").astype(np.float32)


def still_runs(state: np.ndarray, fps: float, still_speed: float, min_s: float = 0.3) -> list[tuple[int, int]]:
    """[start, end) frame ranges where the arm stays nearly still for at least min_s."""
    still = joint_speed(state, fps) <= still_speed
    runs, start = [], None
    for i, s in enumerate(np.append(still, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_s * fps:
                runs.append((start, i))
            start = None
    return runs


def inspection_frames(ep: Episode, inspect_pose: np.ndarray | None, tolerance: float, still_speed: float) -> np.ndarray:
    """Frame indices (within the episode) where the brick is held at the inspection camera.

    With a recorded inspect pose: still frames within `tolerance` of it. Without
    one: the longest still stretch in the middle of the episode (after the
    pick, before the drop), which is where demonstrations pause to inspect.
    """
    speed = joint_speed(ep.state, ep.fps)
    if inspect_pose is not None:
        near = np.abs(ep.state[:, :5] - inspect_pose[:5]).max(axis=1) <= tolerance
        return np.flatnonzero(near & (speed <= still_speed))
    T = len(ep.state)
    runs = [(a, b) for a, b in still_runs(ep.state, ep.fps, still_speed) if a > 0.2 * T and b < 0.9 * T]
    if not runs:
        return np.array([], int)
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return np.arange(a, b)
