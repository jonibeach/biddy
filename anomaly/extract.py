"""Pull inspection-camera crops out of recorded demonstrations for Anomalib.

Every demo already holds the brick in front of the inspection camera, and its
task text says which bowl it went to, so the demos double as labeled
anomaly data with no extra recording:

    ~/physical-ai-studio/application/backend/.venv/bin/python anomaly/extract.py latest

writes data/anomaly/good/*.png and data/anomaly/defective/*.png (the task
containing "defective" -> defective, the rest -> good), cropped to
inspection.roi from station.yaml. Re-running replaces the extracted crops but
keeps any other files in those folders (e.g. from anomaly/capture.py).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cell import poses as poses_mod  # noqa: E402
from cell.anomaly import crop  # noqa: E402
from cell.dataset import Dataset, inspection_frames, resolve  # noqa: E402

PREFIX = "ds_"  # extracted files start with this, so a re-run only replaces its own


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="latest")
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--out", default="data/anomaly", type=Path)
    ap.add_argument("--per-episode", type=int, default=5, help="crops per episode, spread over the pause")
    ap.add_argument("--defect-word", default="defective", help="task text containing this -> defective")
    ap.add_argument("--no-crop", action="store_true", help="save the full frame (to pick the ROI from)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    insp = cfg["inspection"]
    roi = None if args.no_crop else insp.get("roi")
    if roi is None and not args.no_crop:
        print("note: inspection.roi is not set; saving full frames. Set it with anomaly/roi.py, then re-run.")
    root = resolve(args.dataset, cfg["studio"]["db"])
    ds = Dataset(root)
    inspect_pose = poses_mod.load(cfg["poses"]).get("inspect")

    for label in ("good", "defective"):
        (args.out / label).mkdir(parents=True, exist_ok=True)
        for old in (args.out / label).glob(f"{PREFIX}*.png"):
            old.unlink()

    counts = {"good": 0, "defective": 0}
    skipped = []
    for ep in ds.episodes():
        if "|" in ep.task:
            skipped.append((ep.index, "several tasks"))
            continue
        idx = inspection_frames(ep, inspect_pose, insp["pose_tolerance"], insp["still_speed"])
        if len(idx) == 0:
            skipped.append((ep.index, "no pause at inspection"))
            continue
        label = "defective" if args.defect_word in ep.task.lower() else "good"
        # Spread picks over the pause, skipping its edges where motion blur is likeliest.
        inner = idx[len(idx) // 5: len(idx) - len(idx) // 5] if len(idx) >= 5 else idx
        picks = np.unique(np.linspace(0, len(inner) - 1, min(args.per_episode, len(inner))).round().astype(int))
        for j in picks:
            rgb = crop(ds.frame(int(ep.global_index[inner[j]]), insp["camera"]), roi)
            name = f"{PREFIX}{root.name[:8]}_ep{ep.index:04d}_f{int(inner[j]):05d}.png"
            cv2.imwrite(str(args.out / label / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            counts[label] += 1
    print(f"wrote {counts['good']} good, {counts['defective']} defective crops to {args.out}")
    for ep, why in skipped:
        print(f"  skipped episode {ep}: {why}")


if __name__ == "__main__":
    main()
