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


def detect(frame, detector):
    corners, ids, _ = detector.detectMarkers(frame)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            found[int(i)] = c.reshape(4, 2).mean(axis=0).tolist()
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="camera index")
    ap.add_argument("--image", type=str, help="image file instead of camera")
    ap.add_argument("--name", required=True, help="which camera this is: top, inspection or wrist")
    ap.add_argument("--save-reference", action="store_true")
    args = ap.parse_args()
    REF_FILE = Path(__file__).with_name(f"reference_{args.name}.json")

    if args.image:
        frame = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(args.camera)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"No frame from camera {args.camera}")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    found = detect(frame, detector)

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
