"""AprilTag guard: has a camera or a fixture moved since the demos were recorded?

The tags (calibration/make_tags.py) sit on the table, the arm base, the
inspection stand, the bowls and in the inspection camera's background. Their
pixel centres at recording time are in calibration/reference_<camera>.json
(calibration/tag_check.py --save-reference). The cell looks at them with the
arm at home before every brick (cell/run.py, station.yaml `tags`):

  * a tag further from its reference than its tolerance keeps the cell from
    picking until it is back (on_moved: pause) or is only reported (warn);
  * for a camera listed under `correct`, a camera that moved is not a
    problem: its inspection crop boxes move with the tags, so each Anomalib
    model sees the brick where it was trained to.

Tags that are not visible are ignored: a brick or the arm may cover them.
The crop boxes move by whole pixels and the frame is never resampled: the
boxes reach the edge of the image, and a warped frame has made-up pixels
there, which the models score as defects. A small turn of the camera is
therefore left in the crop, and a larger one (MAX_TURN_DEG) is a problem.
A camera that was pushed sideways shifts the brick (close) more than the
tags (far), so some error is left. The references are pixel positions in
Studio's 640x480 frames, the size the cell reads its cameras at.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from calibration.make_tags import ASSIGNMENTS

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "calibration"
MAX_TURN_DEG = 3.0  # moving a crop box cannot undo a turn; a 360 px crop is off by 9 px at its ends here
MAX_SCALE = 0.03    # nor a change of size (camera pushed closer or further)


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    # Wider threshold windows and a lower constant than the defaults: tags with
    # tape on the white margin or glare on the table are otherwise missed.
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 85
    params.adaptiveThreshWinSizeStep = 8
    params.adaptiveThreshConstant = 3
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect(frames, detector) -> dict[int, list[float]]:
    """Median pixel centre per tag over all frames in which it was detected."""
    seen = {}
    for frame in frames:
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                seen.setdefault(int(i), []).append(c.reshape(4, 2).mean(axis=0))
    return {i: np.median(np.array(pts), axis=0).tolist() for i, pts in seen.items()}


def load_reference(camera: str) -> dict[int, list[float]]:
    path = REFERENCE_DIR / f"reference_{camera}.json"
    if not path.exists():
        raise SystemExit(f"no {path}; save it with calibration/tag_check.py --studio --name {camera} "
                         f"--save-reference, or take {camera!r} out of tags.cameras in station.yaml")
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def fit_camera(src: np.ndarray, dst: np.ndarray, tolerance: float) -> np.ndarray | None:
    """2x3 shift + rotation + scale that most tags agree on, taking src points to dst.

    Most tags are on things that stay put, so the movement they share is the
    camera's; a tag on something that moved by itself is left out of the fit
    (RANSAC). Four numbers fitted to many centres also average out detection
    jitter, which a full homography through four centres would follow.
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if len(src) == 1:
        tx, ty = dst[0] - src[0]
        return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=tolerance)
    return matrix


def follow(views: dict, matrices: dict[str, np.ndarray]) -> dict:
    """inspection.views with the crop box of every view on a moved camera moved along.

    matrices: camera -> 2x3 taking current pixels to reference pixels. Each
    box keeps its size and moves by the whole-pixel shift of its centre.
    """
    if not matrices:
        return views
    out = {}
    for name, v in views.items():
        m, roi = matrices.get(v.get("camera")), v.get("roi")
        if m is not None and roi:
            x, y, w, h = roi
            cx, cy = cv2.invertAffineTransform(m) @ [x + w / 2, y + h / 2, 1.0]
            v = {**v, "roi": [int(round(cx - w / 2)), int(round(cy - h / 2)), w, h]}
        out[name] = v
    return out


@dataclass
class CameraCheck:
    shift: float = 0.0                 # how far the tags are from their reference, px
    matrix: np.ndarray | None = None   # current -> reference pixels, when the crop boxes are to follow
    problem: str = ""                  # something moved that the cell cannot make up for
    note: str = ""                     # worth logging, no action needed


def check_camera(camera: str, found: dict[int, list[float]], ref: dict[int, list[float]], tolerance: float,
                 tolerances: dict[str, float] | None = None, correct: bool = False,
                 max_correction: float = 40.0) -> CameraCheck:
    """Compare one camera's detected tag centres with its reference.

    The movement most tags share is the camera's (fit_camera). An object
    whose tags are still off after taking that out moved by itself.
    """
    tolerances = tolerances or {}
    ids = sorted(i for i in ref if i in found)
    if not ids:
        return CameraCheck(note=f"no tags visible on the {camera} camera")
    cur, want = np.array([found[i] for i in ids]), np.array([ref[i] for i in ids])
    names = [ASSIGNMENTS.get(i, f"tag {i}") for i in ids]

    def off(distances) -> dict[str, float]:
        """object -> median distance of its tags, for objects beyond their tolerance."""
        per = {n: float(np.median([d for m, d in zip(names, distances) if m == n])) for n in dict.fromkeys(names)}
        return {n: d for n, d in per.items() if d > tolerances.get(n, tolerance)}

    shifts = np.hypot(*(cur - want).T)
    result = CameraCheck(shift=float(shifts.max()))
    moved = off(shifts)
    if not moved:
        return result

    matrix = fit_camera(cur, want, tolerance)
    if matrix is None:
        result.problem = f"{camera} camera: " + ", ".join(f"{n} moved {d:.0f} px" for n, d in moved.items())
        return result
    fitted = cur @ matrix[:, :2].T + matrix[:, 2]
    alone = off(np.hypot(*(fitted - want).T))        # still off once the camera's movement is taken out
    explained = [n for n in moved if n not in alone]
    # A correctable camera's tags are its own fixed background (one object);
    # otherwise two objects must agree before the camera is blamed.
    camera_moved = bool(explained) and (correct or len(explained) >= 2)
    if not camera_moved:
        alone = moved
    by_camera = float(np.median(np.hypot(*(fitted - cur).T)))
    turn = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    problems = [f"{n} moved {d:.0f} px" for n, d in alone.items()]
    if camera_moved and not correct:
        problems.insert(0, f"the camera moved {by_camera:.0f} px")
    elif camera_moved and by_camera > max_correction:
        problems.insert(0, f"the camera moved {by_camera:.0f} px, more than the {max_correction:.0f} px the crops can follow")
    elif camera_moved and abs(turn) > MAX_TURN_DEG:
        problems.insert(0, f"the camera turned {abs(turn):.1f} deg, more than the crops can follow")
    elif camera_moved and abs(scale - 1) > MAX_SCALE:
        problems.insert(0, f"the camera's image changed size by {abs(scale - 1):.0%}, more than the crops can follow")
    if problems:
        result.problem = f"{camera} camera: " + ", ".join(problems)
    else:
        result.shift, result.matrix = by_camera, matrix
        result.note = f"{camera} camera moved {by_camera:.0f} px, crops follow the tags"
    return result


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    matrices: dict[str, np.ndarray] = field(default_factory=dict)  # camera -> current to reference pixels
    shift: float = 0.0  # largest tag shift the crops follow, px


class TagGuard:
    """Checks the tags of the configured cameras against their references."""

    def __init__(self, cfg: dict):
        self.cameras = list(cfg.get("cameras") or [])
        self.correct = set(cfg.get("correct") or [])
        unknown = self.correct - set(self.cameras)
        if unknown:
            raise SystemExit(f"tags.correct lists {sorted(unknown)}, which are not in tags.cameras")
        self.tolerance = float(cfg.get("tolerance_px", 5.0))
        self.tolerances = {k: float(v) for k, v in (cfg.get("tolerances") or {}).items()}
        self.max_correction = float(cfg.get("max_correction_px", 40.0))
        self.frames = int(cfg.get("frames", 5))
        self.on_moved = str(cfg.get("on_moved", "pause"))
        if self.on_moved not in ("pause", "warn"):
            raise SystemExit(f"tags.on_moved must be pause or warn, not {self.on_moved!r}")
        self.refs = {cam: load_reference(cam) for cam in self.cameras}
        self.detector = make_detector()

    @classmethod
    def from_config(cls, cfg: dict | None) -> "TagGuard | None":
        """None when station.yaml has no `tags` section, no cameras in it, or enabled: false."""
        if not cfg or not cfg.get("enabled", True) or not cfg.get("cameras"):
            return None
        return cls(cfg)

    def check_frames(self, frames: dict[str, list[np.ndarray]]) -> Report:
        """frames: camera -> a few frames taken with the arm at home."""
        report = Report()
        for cam in self.cameras:
            r = check_camera(cam, detect(frames[cam], self.detector), self.refs[cam], self.tolerance,
                             self.tolerances, cam in self.correct, self.max_correction)
            if r.problem:
                report.problems.append(r.problem)
            if r.note:
                report.notes.append(r.note)
            if r.matrix is not None:
                report.matrices[cam] = r.matrix
                report.shift = max(report.shift, r.shift)
        return report

    def check(self, cameras: dict) -> Report:
        """Read a few frames from each camera (arm at home) and check them."""
        frames: dict[str, list[np.ndarray]] = {cam: [] for cam in self.cameras}
        for _ in range(self.frames):
            for cam in self.cameras:
                frames[cam].append(cameras[cam].read().data)
        return self.check_frames(frames)
