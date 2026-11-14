"""Pick the inspection crop (ROI) by dragging a box, and save it to station.yaml.

    python anomaly/roi.py --view inspection data/anomaly/inspection/good/<full frame>.png
    python anomaly/roi.py --view wrist --camera     # grab a live frame from that view's camera

inspection: drag a box around where the brick sits at the inspection spot,
with a margin for the tilted and rolled holds. wrist: a box around the brick
between the jaw tips during the hover. Enter accepts, c cancels. Get full
frames first with `anomaly/extract.py --no-crop` if Studio holds the cameras.
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def live_frame(cfg, camera: str) -> "cv2.Mat":
    from cell.hardware import build_cameras

    cam = build_cameras({**cfg, "studio": {**cfg["studio"], "cameras": [camera]}})
    cam = next(iter(cam.values()))
    cam.connect()
    try:
        for _ in range(15):  # let auto-exposure settle
            frame = cam.read()
        return cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)
    finally:
        cam.disconnect()


def write_roi(config: Path, view: str, roi: list[int]):
    """Replace the roi: line inside inspection.views.<view>, keeping comments and layout."""
    lines = config.read_text().splitlines(keepends=True)
    path = ["inspection", "views", view]
    depth, indent = 0, -1
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        ind = len(line) - len(stripped)
        if depth == len(path):
            if ind <= indent:
                break  # left the view's block without finding roi
            if stripped.startswith("roi:"):
                comment = re.search(r"\s+#.*$", line.rstrip("\n"))
                lines[i] = f"{line[:ind]}roi: {roi}{comment.group(0) if comment else ''}\n"
                config.write_text("".join(lines))
                return
        elif ind > indent and stripped.split(":")[0] == path[depth]:
            depth, indent = depth + 1, ind
    raise SystemExit(f"no inspection.views.{view}.roi line in {config}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("--camera", action="store_true", help="grab a live frame from the view's camera")
    ap.add_argument("--view", default="inspection", help="inspection or wrist (inspection.views in station.yaml)")
    ap.add_argument("--config", default="station.yaml", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.view not in cfg["inspection"]["views"]:
        raise SystemExit(f"no view {args.view!r}; have {list(cfg['inspection']['views'])}")
    if args.camera:
        img = live_frame(cfg, cfg["inspection"]["views"][args.view]["camera"])
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
    else:
        raise SystemExit("give an image path or --camera")
    x, y, w, h = cv2.selectROI(f"{args.view} ROI (Enter to accept)", img, showCrosshair=True)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise SystemExit("no box selected")
    roi = [int(x), int(y), int(w), int(h)]
    write_roi(args.config, args.view, roi)
    print(f"inspection.views.{args.view}.roi = {roi} written to {args.config}")


if __name__ == "__main__":
    main()
