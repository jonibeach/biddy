"""Detect AprilTags from a camera and check whether the fixed parts moved.

Usage:
  python tag_check.py --studio --name top --save-reference     # once, after setup
  python tag_check.py --studio --name top                      # each session
  python tag_check.py --camera 0 --name top                    # camera not held by Studio
  python tag_check.py --image a.jpg b.jpg --name top           # saved images instead

--studio reads Physical AI Studio's shared camera feed as one more viewer,
at the size Studio records with (640x480). It works while Studio is running,
also during recording, and needs Studio's backend venv:
  ~/physical-ai-studio/application/backend/.venv/bin/python tag_check.py --studio ...
The saved references were made with --studio, so check with --studio too:
pixel positions depend on capture size.

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

TOLERANCE_FRAC = 8.0 / 1920  # allowed center shift, as a fraction of frame width
WIDTH, HEIGHT = 1920, 1080  # requested capture size; the camera may give less
WARMUP_FRAMES = 15  # let auto-exposure settle before detecting
DETECT_FRAMES = 10  # a tag counts as visible if any of these frames shows it
STUDIO = "localhost:7860"
STUDIO_PROJECT = "biddy"
STUDIO_BACKEND = Path("~/physical-ai-studio/application/backend").expanduser()
MIN_TOLERANCE_PX = 4.0  # detection jitter at 640x480 is up to about 3 px


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


def grab_studio(name):
    """Frames of the Studio project camera called `name`, read from Studio's
    shared camera feed as one more viewer. Works while a recording session is
    open, and never takes the camera from Studio."""
    import sys
    import time
    import urllib.request

    sys.path.insert(0, str(STUDIO_BACKEND / "src"))
    from pydantic import TypeAdapter
    from schemas.project_camera import Camera
    from utils.camera_factory import build_shared_camera

    def get(path):
        return json.load(urllib.request.urlopen(f"http://{STUDIO}{path}"))

    project = next(p for p in get("/api/projects") if p["name"] == STUDIO_PROJECT)
    camera = next((c for c in get(f"/api/projects/{project['id']}/cameras") if c["name"] == name), None)
    if camera is None:
        raise SystemExit(f"Studio project {STUDIO_PROJECT!r} has no camera called {name!r}")
    shared = build_shared_camera(
        config=TypeAdapter(Camera).validate_python(camera), validate_on_connect=False, overwrite_settings=False
    )
    shared.connect()
    frames, last, deadline = [], None, time.monotonic() + 10
    try:
        while len(frames) < DETECT_FRAMES:
            if time.monotonic() > deadline:
                raise SystemExit(f"No frames from Studio camera {name!r}; is Studio running?")
            frame = shared.read_latest()
            if frame.sequence != last:
                last = frame.sequence
                frames.append(cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR))
            time.sleep(0.02)
    finally:
        shared.disconnect()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="camera index")
    ap.add_argument("--image", nargs="+", help="image files instead of a camera")
    ap.add_argument("--studio", action="store_true", help="read the camera named --name from Studio")
    ap.add_argument("--name", required=True, help="which camera this is: top, inspection or wrist")
    ap.add_argument("--save-reference", action="store_true")
    args = ap.parse_args()
    REF_FILE = Path(__file__).with_name(f"reference_{args.name}.json")

    if args.image:
        frames = [cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR) for f in args.image]
    elif args.studio:
        frames = grab_studio(args.name)
    else:
        frames = grab(args.camera)
    tolerance = max(MIN_TOLERANCE_PX, TOLERANCE_FRAC * frames[0].shape[1])

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
            if d > tolerance:
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
