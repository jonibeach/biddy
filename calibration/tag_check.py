"""Detect AprilTags from a camera and check whether the fixed parts moved.

Usage:
  python tag_check.py --camera 0 --name top --save-reference   # once, after setup
  python tag_check.py --camera 0 --name top                    # each session
  python tag_check.py --image shot.png --name top              # test on a saved image

Reports the pixel center of each tag, grouped by object, and compares it
to reference_<name>.json. Reading the result for a fixed camera:
  every group shifted together -> that camera moved
  one group shifted            -> that object moved
The wrist camera moves with the arm, so only check it with the arm parked
in the same home pose each time.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from make_tags import ASSIGNMENTS

TOLERANCE_PX = 8.0  # allowed center shift before a tag counts as moved
WIDTH, HEIGHT = 1920, 1080  # requested capture size; the camera may give less
WARMUP_FRAMES = 15  # let auto-exposure settle before detecting
DETECT_FRAMES = 10  # a tag counts as visible if any of these frames shows it


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


def detect(frames, detector):
    """Median center per tag over all frames in which it was detected."""
    seen = {}
    for frame in frames:
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                seen.setdefault(int(i), []).append(c.reshape(4, 2).mean(axis=0))
    return {i: np.median(np.array(pts), axis=0).tolist() for i, pts in seen.items()}


def grab(camera):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    frames = []
    for _ in range(WARMUP_FRAMES + DETECT_FRAMES):
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    frames = frames[WARMUP_FRAMES:]
    if not frames:
        raise SystemExit(f"No frame from camera {camera}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="camera index")
    ap.add_argument("--image", type=str, help="image file instead of camera")
    ap.add_argument("--name", required=True, help="which camera this is: top, inspection or wrist")
    ap.add_argument("--save-reference", action="store_true")
    args = ap.parse_args()
    REF_FILE = Path(__file__).with_name(f"reference_{args.name}.json")

    if args.image:
        frames = [cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)]
    else:
        frames = grab(args.camera)

    found = detect(frames, make_detector())

    print("Visible tags by object:")
    for name in sorted(set(ASSIGNMENTS.values())):
        ids = sorted(i for i in found if ASSIGNMENTS.get(i) == name)
        print(f"  {name:18s} {ids if ids else 'NONE VISIBLE'}")
    unknown = sorted(i for i in found if i not in ASSIGNMENTS)
    if unknown:
        print(f"  unknown IDs: {unknown}")

    if args.save_reference:
        REF_FILE.write_text(json.dumps(found, indent=2))
        print(f"Saved reference for {len(found)} tags to {REF_FILE.name}")
        return

    if not REF_FILE.exists():
        print(f"No {REF_FILE.name} yet; run with --save-reference first.")
        return
    ref = {int(k): v for k, v in json.loads(REF_FILE.read_text()).items()}
    moved = []
    for i, (x, y) in found.items():
        if i in ref:
            d = float(np.hypot(x - ref[i][0], y - ref[i][1]))
            if d > TOLERANCE_PX:
                moved.append((i, ASSIGNMENTS.get(i, "?"), d))
    missing = sorted(i for i in ref if i not in found)
    if moved:
        print("MOVED:")
        for i, name, d in moved:
            print(f"  ID {i} ({name}) shifted {d:.1f} px")
    if missing:
        print(f"Not visible this time: {missing}")
    if not moved:
        print("No visible tag moved beyond tolerance.")


if __name__ == "__main__":
    main()
