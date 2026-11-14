"""Pick the inspection crop (ROI) by dragging a box, and save it to station.yaml.

    python anomaly/roi.py data/anomaly/good/some_full_frame.png
    python anomaly/roi.py --camera     # grab a live frame from the inspection camera

Drag a box around where the brick sits when held at the inspection pose, with
a little margin, then press Enter (c cancels). Get a full frame first with
`anomaly/extract.py --no-crop` if Studio is holding the camera.
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def live_frame(cfg) -> "cv2.Mat":
    from cell.hardware import build_cameras

    cam = build_cameras({**cfg, "studio": {**cfg["studio"], "cameras": [cfg["inspection"]["camera"]]}})
    cam = next(iter(cam.values()))
    cam.connect()
    try:
        for _ in range(15):  # let auto-exposure settle
            frame = cam.read()
        return cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)
    finally:
        cam.disconnect()


def write_roi(config: Path, roi: list[int]):
    text = config.read_text()
    new, n = re.subn(r"(?m)^(\s*roi:)\s*[^#\n]*?(\s*#.*)?$", lambda m: f"{m.group(1)} {roi}{m.group(2) or ''}", text, count=1)
    if n != 1:
        raise SystemExit(f"no 'roi:' line in {config}")
    config.write_text(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--config", default="station.yaml", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.camera:
        img = live_frame(cfg)
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
    else:
        raise SystemExit("give an image path or --camera")
    x, y, w, h = cv2.selectROI("inspection ROI (Enter to accept)", img, showCrosshair=True)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise SystemExit("no box selected")
    roi = [int(x), int(y), int(w), int(h)]
    write_roi(args.config, roi)
    print(f"inspection.roi = {roi} written to {args.config}")


if __name__ == "__main__":
    main()
