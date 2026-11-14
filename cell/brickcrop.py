"""Find the brick in an inspection frame and crop around it.

Fixed crop boxes failed on the real demos: the brick is held near or far, left
or right, so a fixed box often cut it in half or showed mostly gripper, and
Anomalib learned framing instead of marks. Instead, the brick is found by:

  * what changed against the empty view (the same camera with the arm at home),
  * minus the gripper by colour: the pink jaws, the orange camera mount and
    the black servo,
  * the largest remaining blob, inside the view's search box (roi).

The crop is a square around that blob with a margin, so every crop shows one
brick at the same scale, whatever its colour (green, lavender, white, light
blue) or distance. Used for both offline crops (anomaly/extract.py) and the
live cell (cell/run.py).
"""
from __future__ import annotations

import cv2
import numpy as np

DIFF = 35          # per-pixel change against the empty view (0-255) that counts as "something there"
MIN_AREA = 600     # px; smaller blobs are noise or a sliver of brick at the frame edge
MARGIN = 0.15
MASK_GREY = 128    # fill outside the brick when a view sets mask: true      # of the box size, on every side


def brick_mask(rgb: np.ndarray, background: np.ndarray) -> np.ndarray:
    """uint8 mask (255 = brick candidate) of the same size as rgb."""
    if background.shape != rgb.shape:
        background = cv2.resize(background, (rgb.shape[1], rgb.shape[0]))
    changed = cv2.absdiff(rgb, background).max(axis=2) > DIFF
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pink = ((h >= 140) | (h <= 8)) & (s > 70)         # gripper jaws (OpenCV hue 0-179)
    orange = (h > 8) & (h < 22) & (s > 120)           # camera mount
    dark = v < 60                                     # servo, cables
    mask = (changed & ~pink & ~orange & ~dark).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def _largest_blob(mask: np.ndarray):
    """((x, y, w, h), labels == k) of the largest blob, or None if it is too small."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[k]
    if area < MIN_AREA:
        return None
    return (int(x), int(y), int(w), int(h)), labels == k


def elongation(rgb: np.ndarray, background: np.ndarray, roi: list[int] | None = None) -> float | None:
    """Long side / short side of the tightest rotated box around the brick, or None.

    Shape, not colour, tells a long brick from a standard one: the brick is
    held still at the inspection spot with the jaws masked out, so a 2x2 Duplo
    gives a squat blob and a long brick held by one end a stretched one. The
    rotated box makes it independent of how far the wrist is rolled.
    """
    H, W = rgb.shape[:2]
    x0, y0, w0, h0 = roi if roi else (0, 0, W, H)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + w0), min(H, y0 + h0)
    blob = _largest_blob(brick_mask(rgb[y0:y1, x0:x1], background[y0:y1, x0:x1]))
    if blob is None:
        return None
    pts = cv2.findNonZero(blob[1].astype(np.uint8))
    (_, _), (a, b), _ = cv2.minAreaRect(pts)
    return float(max(a, b) / max(1.0, min(a, b)))


def brick_box(rgb: np.ndarray, background: np.ndarray, roi: list[int] | None = None) -> list[int] | None:
    """[x, y, w, h] of a square around the brick in full-frame coordinates, or None."""
    H, W = rgb.shape[:2]
    x0, y0, w0, h0 = roi if roi else (0, 0, W, H)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + w0), min(H, y0 + h0)
    mask = brick_mask(rgb[y0:y1, x0:x1], background[y0:y1, x0:x1])
    blob = _largest_blob(mask)
    if blob is None:
        return None
    x, y, w, h = blob[0]
    side = int(max(w, h) * (1 + 2 * MARGIN))
    cx, cy = x0 + x + w // 2, y0 + y + h // 2
    bx = int(np.clip(cx - side // 2, 0, max(0, W - side)))
    by = int(np.clip(cy - side // 2, 0, max(0, H - side)))
    side = min(side, W, H)
    return [bx, by, side, side]


def brick_crop(rgb: np.ndarray, background: np.ndarray, roi: list[int] | None = None,
               mask_background: bool = False) -> np.ndarray | None:
    """Square crop around the brick, or None.

    mask_background: paint everything outside the brick's outline flat grey, so
    the anomaly model compares brick surface with brick surface instead of the
    cardboard, jaws and mount behind it, which change from hold to hold.
    """
    box = brick_box(rgb, background, roi)
    if box is None:
        return None
    x, y, w, h = box
    out = rgb[y:y + h, x:x + w]
    if not mask_background:
        return out
    H, W = rgb.shape[:2]
    rx, ry, rw, rh = roi if roi else (0, 0, W, H)
    rx, ry = max(0, rx), max(0, ry)
    rx1, ry1 = min(W, rx + rw), min(H, ry + rh)
    blob = _largest_blob(brick_mask(rgb[ry:ry1, rx:rx1], background[ry:ry1, rx:rx1]))
    full = np.zeros((H, W), np.uint8)
    full[ry:ry1, rx:rx1] = blob[1].astype(np.uint8) * 255
    # A little margin so the brick's own edges (where chips and marks show) stay in.
    full = cv2.dilate(full, np.ones((9, 9), np.uint8))
    keep = full[y:y + h, x:x + w] > 0
    out = out.copy()
    out[~keep] = MASK_GREY
    return out
