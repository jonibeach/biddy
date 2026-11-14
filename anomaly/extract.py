"""Pull labeled crops for Anomalib out of recorded demos, one folder per camera view.

Every demo holds the brick in front of the inspection camera (back, bottom,
both sides) and hovers over it first (the wrist camera sees top and front);
its task text says which bowl it went to. So the demos double as anomaly data:

    ~/physical-ai-studio/application/backend/.venv/bin/python anomaly/extract.py latest

writes data/anomaly/<view>/good/*.png and data/anomaly/<view>/defective/*.png
for the views in station.yaml (inspection, wrist), cropped to each view's roi.
Holds are found the same way the live cell finds them (cell/inspection.py).

A defective brick has its mark on one face only, so most crops under
defective/ show a clean face. That is expected: training uses good/ only, and
anomaly/train.py picks the threshold per episode from its worst crop.
File names carry the episode (…_ep0004_h2_…) for that. Re-running replaces
the extracted crops but keeps other files in those folders.
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cell import poses as poses_mod  # noqa: E402
from cell.dataset import Dataset, resolve  # noqa: E402
from cell.inspection import INSPECTION, find_holds, view_crop  # noqa: E402

PREFIX = "ds_"  # extracted files start with this, so a re-run only replaces its own


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="latest")
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--out", default="data/anomaly", type=Path)
    ap.add_argument("--per-hold", type=int, default=3, help="crops per hold, spread over it")
    ap.add_argument("--defect-word", default="defective", help="task text containing this -> defective")
    ap.add_argument("--no-crop", action="store_true", help="save full frames (to pick the ROIs from)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    insp = cfg["inspection"]
    views = insp["views"]
    if args.no_crop:
        views = {name: {**v, "roi": None} for name, v in views.items()}
    for name, v in views.items():
        if v.get("roi") is None and not args.no_crop:
            print(f"note: {name} roi is not set; saving full frames. Set it with ./biddy roi --view {name}, then re-run.")
    root = resolve(args.dataset, cfg["studio"]["db"])
    ds = Dataset(root)
    home = poses_mod.load(cfg["poses"]).get("home")

    for name in views:
        for label in ("good", "defective"):
            folder = args.out / name / label
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob(f"{PREFIX}*.png"):
                old.unlink()

    counts: dict[tuple[str, str], int] = {}
    skipped = []
    for ep in ds.episodes():
        if "|" in ep.task:
            skipped.append((ep.index, "several tasks"))
            continue
        label = "defective" if args.defect_word in ep.task.lower() else "good"

        def frame(view: str, i: int):
            return view_crop(ds.frame(int(ep.global_index[i]), views[view]["camera"]), views[view])

        holds = find_holds(ep.state, ep.fps, lambda i: frame(INSPECTION, i), insp, home)
        if not any(h.kind == INSPECTION for h in holds):
            skipped.append((ep.index, "no inspection hold found"))
            continue
        for h_i, hold in enumerate(holds):
            if hold.kind not in views:
                continue
            for i in hold.frames(args.per_hold):
                rgb = frame(hold.kind, int(i))
                name = f"{PREFIX}{root.name[:8]}_ep{ep.index:04d}_h{h_i}_f{int(i):05d}.png"
                cv2.imwrite(str(args.out / hold.kind / label / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                counts[(hold.kind, label)] = counts.get((hold.kind, label), 0) + 1
        print(f"episode {ep.index}: {label}, holds " + ", ".join(f"{h.kind}@{h.start}" for h in holds))
    for (view, label), n in sorted(counts.items()):
        print(f"wrote {n} {label} crops to {args.out / view / label}")
    for ep, why in skipped:
        print(f"  skipped episode {ep}: {why}")


if __name__ == "__main__":
    main()
