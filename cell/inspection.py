"""Find the inspection holds in a demo or a live run, and turn them into a verdict.

The recording protocol (docs/data_collection.md) shows each brick several
times: a short hover over it (the wrist camera sees the top and front faces),
then holds in front of the inspection camera, with a regrasp in between. The
brick is held at fixed spots in the inspection image, one per kind of face
(the station uses the centre for back and sides, the bottom right for the
bottom), so each spot is its own "view" with its own crop and Anomalib model
(station.yaml inspection.views). A defective brick has its mark on one face
only, so the verdict is the worst face: defective if any hold scores at or
above its view's threshold.

A "hold" is a stretch where the arm, gripper included, is still and away from
home. It belongs to the inspection view whose crop differs most from the empty
view (the brick is there), if that difference reaches presence_diff; a still
stretch before the first inspection hold is the hover, scored on the hover
view (the wrist camera). Everything else (regrasp, bowl) is ignored. The same
rule runs offline on recorded demos (anomaly/extract.py) and live in the cell
(cell/run.py), so each model is trained on the frames it will judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from cell.anomaly import crop


def split_views(views: dict) -> tuple[list[str], str | None]:
    """(inspection view names, hover view name or None) from inspection.views."""
    hover = [name for name, v in views.items() if v.get("hover")]
    if len(hover) > 1:
        raise ValueError(f"only one hover view allowed, got {hover}")
    return [name for name in views if name not in hover], (hover[0] if hover else None)


def presence(rgb_crop: np.ndarray, background_crop: np.ndarray) -> float:
    """Mean absolute pixel difference (0-255) between a crop and the empty view."""
    if rgb_crop.shape != background_crop.shape:
        background_crop = cv2.resize(background_crop, (rgb_crop.shape[1], rgb_crop.shape[0]))
    return float(np.mean(cv2.absdiff(rgb_crop, background_crop)))


def near(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    """Every body joint (gripper ignored) within tolerance."""
    return float(np.max(np.abs(np.asarray(a)[:5] - np.asarray(b)[:5]))) <= tolerance


def at_pose(state: np.ndarray, view_cfg: dict, default_tolerance: float) -> float | None:
    """Distance to the nearest of the view's arm poses if within tolerance, else None."""
    poses = view_cfg.get("poses") or []
    tol = float(view_cfg.get("pose_tolerance", default_tolerance))
    dists = [float(np.max(np.abs(np.asarray(state)[:5] - np.asarray(p, np.float32)[:5]))) for p in poses]
    best = min(dists, default=None)
    return best if best is not None and best <= tol else None


def hold_kind(state: np.ndarray, home: np.ndarray, presences: dict[str, float], seen_inspection: bool,
              cfg: dict, hover_view: str | None) -> str | None:
    """Which view a still stretch at `state` belongs to, or None.

    An inspection view with recorded arm poses (inspection.views.<v>.poses,
    found with scripts/find_spots.py) claims holds at those poses, provided its
    crop shows something (presence). The arm also wanders into the inspection
    camera's view at other times, so pixel change alone over-counts holds;
    views without poses fall back to it. presences: view -> difference of its
    crop from the empty view.
    """
    if near(state, home, float(cfg["home_tolerance"])):
        return None
    views = cfg["views"]
    by_pose = {v: d for v in presences if (d := at_pose(state, views[v], float(cfg.get("pose_tolerance", 10.0)))) is not None}
    threshold = float(cfg["presence_diff"])
    if by_pose:
        view = min(by_pose, key=by_pose.get)
        return view if presences[view] >= threshold else None
    unposed = {v: p for v, p in presences.items() if not views[v].get("poses")}
    if unposed:
        view, p = max(unposed.items(), key=lambda kv: kv[1])
        if p >= threshold:
            return view
    if not seen_inspection:
        return hover_view
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
    hover_view: str | None = None
    scores: dict[str, list[float]] = field(default_factory=dict)  # view -> one score per hold

    def add(self, view: str, score: float):
        self.scores.setdefault(view, []).append(score)

    @property
    def holds(self) -> int:
        """Inspection holds scored (the hover not counted)."""
        return sum(len(s) for v, s in self.scores.items() if v != self.hover_view)

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


def find_holds(state: np.ndarray, fps: float, inspection_crops, cfg: dict, hover_view: str | None,
               home: np.ndarray | None = None) -> list[Hold]:
    """Holds in a recorded episode.

    inspection_crops(i) returns {inspection view: cropped RGB} at episode frame i.
    The episode starts at home with the inspection camera's view empty, so
    frame 0 is both the home pose (unless given) and the empty background.
    """
    from cell.dataset import still_runs

    home = state[0] if home is None else home
    backgrounds = inspection_crops(0)
    holds: list[Hold] = []
    for a, b in still_runs(state, fps, float(cfg["still_speed"]), float(cfg.get("hold_min_s", 0.3)), gripper=True):
        mid = (a + b) // 2
        crops = inspection_crops(mid)
        presences = {v: presence(c, backgrounds[v]) for v, c in crops.items()}
        seen = any(h.kind != hover_view for h in holds)
        kind = hold_kind(state[mid], home, presences, seen, cfg, hover_view)
        if kind:
            holds.append(Hold(a, b, kind, presences.get(kind, float("nan"))))
    return holds


def view_crop(rgb: np.ndarray, view_cfg: dict) -> np.ndarray:
    return crop(rgb, view_cfg.get("roi"))


def episode_crops(ds, ep, views: dict, names: list[str]):
    """crops(i) -> {view: cropped RGB} for an episode, decoding each camera once per frame."""

    def crops(i: int) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        out = {}
        for name in names:
            cam = views[name]["camera"]
            if cam not in frames:
                frames[cam] = ds.frame(int(ep.global_index[i]), cam)
            out[name] = view_crop(frames[cam], views[name])
        return out

    return crops
