"""Find the inspection holds in a demo or a live run, and turn them into a verdict.

The recording protocol (docs/data_collection.md) shows each brick several
times: a short hover over it (the wrist camera sees the top and front faces),
then holds in front of the inspection camera for the back, bottom and both
sides, with a regrasp in between. A defective brick has its mark on one face
only, so the verdict is the worst face: defective if any hold scores at or
above its view's threshold.

A "hold" is a stretch where the arm, gripper included, is still and away from home. It is an
inspection hold when the inspection crop differs from the empty view (the
brick is there); a still stretch before the first inspection hold is the
hover, scored on the wrist camera. Everything else (regrasp, bowl) is ignored.
The same rule runs offline on recorded demos (anomaly/extract.py) and live in
the cell (cell/run.py), so the model is trained on the frames it will judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from cell.anomaly import crop

INSPECTION, WRIST = "inspection", "wrist"


def presence(rgb_crop: np.ndarray, background_crop: np.ndarray) -> float:
    """Mean absolute pixel difference (0-255) between a crop and the empty view."""
    if rgb_crop.shape != background_crop.shape:
        background_crop = cv2.resize(background_crop, (rgb_crop.shape[1], rgb_crop.shape[0]))
    return float(np.mean(cv2.absdiff(rgb_crop, background_crop)))


def near(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    """Every body joint (gripper ignored) within tolerance."""
    return float(np.max(np.abs(np.asarray(a)[:5] - np.asarray(b)[:5]))) <= tolerance


def hold_kind(state: np.ndarray, home: np.ndarray, insp_presence: float, seen_inspection: bool,
              cfg: dict) -> str | None:
    """What a still stretch at `state` is: 'inspection', 'wrist' (the hover) or None."""
    if near(state, home, float(cfg["home_tolerance"])):
        return None
    if insp_presence >= float(cfg["presence_diff"]):
        return INSPECTION
    if not seen_inspection:
        return WRIST
    return None


@dataclass
class Hold:
    start: int
    end: int  # exclusive
    kind: str
    presence: float

    def frames(self, n: int) -> np.ndarray:
        """n frame indices spread over the hold, skipping its edges (motion blur)."""
        length = self.end - self.start
        trim = length // 5 if length >= 5 else 0
        a, b = self.start + trim, self.end - trim
        return np.unique(np.linspace(a, b - 1, min(n, b - a)).round().astype(int))


@dataclass
class Verdict:
    """Worst-face decision over all holds of one brick."""

    thresholds: dict[str, float]
    scores: dict[str, list[float]] = field(default_factory=dict)  # view -> one score per hold

    def add(self, view: str, score: float):
        self.scores.setdefault(view, []).append(score)

    @property
    def holds(self) -> int:
        return len(self.scores.get(INSPECTION, []))

    def worst(self) -> tuple[str, float]:
        """(view, score) of the hold furthest above (or closest below) its threshold."""
        best = ("", float("-inf"), float("-inf"))
        for view, scores in self.scores.items():
            for s in scores:
                margin = s - self.thresholds[view]
                if margin > best[2]:
                    best = (view, s, margin)
        return best[0], best[1]

    @property
    def defective(self) -> bool:
        return any(s >= self.thresholds[v] for v, ss in self.scores.items() for s in ss)

    def margin(self) -> float:
        view, score = self.worst()
        return abs(score - self.thresholds[view]) if view else float("inf")


def find_holds(state: np.ndarray, fps: float, inspection_frame, cfg: dict, home: np.ndarray | None = None) -> list[Hold]:
    """Holds in a recorded episode.

    inspection_frame(i) returns the cropped inspection RGB at episode frame i.
    The episode starts at home with the inspection view empty, so frame 0 is
    both the home pose (unless given) and the empty background.
    """
    from cell.dataset import still_runs

    home = state[0] if home is None else home
    background = inspection_frame(0)
    holds: list[Hold] = []
    for a, b in still_runs(state, fps, float(cfg["still_speed"]), float(cfg.get("hold_min_s", 0.3)), gripper=True):
        mid = (a + b) // 2
        p = presence(inspection_frame(mid), background)
        kind = hold_kind(state[mid], home, p, any(h.kind == INSPECTION for h in holds), cfg)
        if kind:
            holds.append(Hold(a, b, kind, p))
    return holds


def view_crop(rgb: np.ndarray, view_cfg: dict) -> np.ndarray:
    return crop(rgb, view_cfg.get("roi"))
