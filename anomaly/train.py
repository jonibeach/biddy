"""Train an Anomalib model on inspection crops and export it for the cell.

Data layout (what anomaly/capture.py writes):

    data/anomaly/<view>/good/*.png        crops from good-brick demos, the more the better
    data/anomaly/<view>/defective/*.png   crops from defective-brick demos; only used to
                                          test and pick the threshold (per episode)

Train one model per camera view (./biddy train-anomaly does all views).

Run with the anomaly/ venv, which has PyTorch's Intel GPU (XPU) build:

    uv sync --project anomaly
    anomaly/.venv/bin/python anomaly/train.py --data data/anomaly/inspection --out models/inspection

Writes <out>/model.xml + model.bin (OpenVINO, fixed [1, 3, S, S] input, FP16
weights), <out>/meta.json, and a score report on the held-out images. With
--int8 it also writes <out>-int8/ quantized with NNCF on the training images.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cell.anomaly import AnomalyModel  # noqa: E402

AUG_PREFIX = "aug_"  # augmented copies of good crops (augment_good)

MODELS = {
    "patchcore": ("anomalib.models", "Patchcore"),
    "padim": ("anomalib.models", "Padim"),
    "efficient_ad": ("anomalib.models", "EfficientAd"),
}


def build_model(name: str, coreset: float):
    import importlib

    module, cls = MODELS[name]
    kwargs = {"coreset_sampling_ratio": coreset} if name == "patchcore" else {}
    return getattr(importlib.import_module(module), cls)(**kwargs)


def heldout_images(datamodule) -> list[tuple[str, int]]:
    """(path, label) of every image the datamodule kept out of training."""
    seen = {}
    for ds in (datamodule.val_data, datamodule.test_data):
        for _, row in ds.samples.iterrows():
            path = str(row["image_path"])
            if not Path(path).name.startswith(AUG_PREFIX):  # copies of training crops would flatter the threshold
                seen[path] = int(row["label_index"])
    return sorted(seen.items())


def score_images(model_dir: Path, images: list[tuple[str, int]], device: str = "CPU"):
    m = AnomalyModel(model_dir, device, threshold=0.5)
    scores = []
    for path, label in images:
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        scores.append((path, label, m(rgb).score))
    return scores


def by_episode(scores) -> list[tuple[str, int, float]]:
    """Collapse crop scores to one per recorded episode: its worst (highest) crop.

    A defective brick has its mark on one face, so only its worst crop is
    expected to score high; that is also how the cell decides live. Crops
    from anomaly/extract.py carry the episode in their name (…_ep0004_h2_…);
    other files count as their own episode.
    """
    import re

    episodes: dict[str, tuple[int, float]] = {}
    for path, label, s in scores:
        m = re.search(r"^(.*_ep\d+)_", Path(path).name)
        key = f"{label}:{m.group(1)}" if m else path
        prev = episodes.get(key)
        episodes[key] = (label, max(s, prev[1]) if prev else s)
    return [(key.split(":", 1)[-1], label, s) for key, (label, s) in episodes.items()]


def pick_threshold(scores) -> tuple[float, str]:
    good = [s for _, label, s in scores if label == 0]
    bad = [s for _, label, s in scores if label == 1]
    if not good or not bad:
        return 0.5, "no held-out good and defective images both; using Anomalib's 0.5"
    hi_good, lo_bad = max(good), min(bad)
    if hi_good < lo_bad:
        return (hi_good + lo_bad) / 2, f"clean split: good <= {hi_good:.3f} < {lo_bad:.3f} <= defective"
    # Overlap: minimize errors, ties broken toward catching defects.
    candidates = sorted(set(good + bad))
    best = min(candidates, key=lambda t: (sum(g >= t for g in good) + sum(b < t for b in bad), t))
    errors = sum(g >= best for g in good) + sum(b < best for b in bad)
    return best, f"scores overlap; {errors}/{len(scores)} held-out images misclassified at {best:.3f}"



def augment_good(good_dir: Path, copies: int):
    """Add rotated/flipped copies of the good crops, so a brick held at another
    orientation does not look anomalous. Patchcore has no rotation invariance
    of its own; it only knows the orientations it was shown.

    Copies are named aug_* and rebuilt on every run. Only good crops are
    augmented, never the held-out defective ones.
    """
    for old in good_dir.glob(f"{AUG_PREFIX}*.png"):
        old.unlink()
    ops = [lambda im: cv2.rotate(im, cv2.ROTATE_180), lambda im: cv2.flip(im, 1),
           lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
           lambda im: cv2.flip(im, 0), lambda im: cv2.transpose(im), lambda im: cv2.flip(cv2.transpose(im), -1)]
    originals = sorted(p for p in good_dir.glob("*.png") if not p.name.startswith(AUG_PREFIX))
    for path in originals:
        img = cv2.imread(str(path))
        for k, op in enumerate(ops[:copies]):
            out = op(img)
            if out.shape[:2] != img.shape[:2]:  # 90-degree turns of a non-square crop: keep the crop's size
                out = cv2.resize(out, (img.shape[1], img.shape[0]))
            cv2.imwrite(str(good_dir / f"{AUG_PREFIX}{k}_{path.name}"), out)
    if copies:
        print(f"augmented {len(originals)} good crops x{copies} (rotations/flips)")


def trainer_device(choice: str) -> dict:
    """Lightning accelerator/strategy for the Intel GPU (XPU), or the CPU."""
    import torch

    if choice == "xpu":
        if not torch.xpu.is_available():
            sys.exit("no Intel GPU visible to PyTorch; use the anomaly/ venv (XPU build) or --accelerator cpu")
        from anomalib.engine import SingleXPUStrategy, XPUAccelerator

        print(f"training on Intel GPU: {torch.xpu.get_device_name(0)}")
        return {"accelerator": XPUAccelerator(), "strategy": SingleXPUStrategy(), "devices": 1}
    print("training on CPU")  # Patchcore/Padim only extract features: seconds on the Core Ultra CPU
    return {"accelerator": "cpu"}


def quantize_int8(fp_xml: Path, calib_images: list[Path], size: int, out_xml: Path):
    """Post-training INT8 quantization with NNCF, calibrated on good images.

    Done here rather than through Anomalib's INT8_PTQ export, which feeds
    calibration images at their original size and fails on a fixed-size model.
    """
    import nncf
    import openvino as ov

    def load(path: Path) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)  # same as AnomalyModel.preprocess
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    model = ov.Core().read_model(fp_xml)
    quantized = nncf.quantize(model, nncf.Dataset(calib_images, load), subset_size=min(300, len(calib_images)))
    ov.save_model(quantized, out_xml)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/anomaly", type=Path)
    ap.add_argument("--out", default="models/anomaly", type=Path)
    ap.add_argument("--model", default="patchcore", choices=sorted(MODELS))
    ap.add_argument("--size", type=int, default=256, help="square model input size")
    ap.add_argument("--epochs", type=int, default=1, help="Patchcore/Padim need 1; EfficientAd ~ 20+")
    ap.add_argument("--int8", action="store_true", help="also export an NNCF INT8 model")
    ap.add_argument("--coreset", type=float, default=0.01,
                    help="Patchcore memory bank as a fraction of all patches. Its greedy selection is the slow "
                         "part (it grows with images x patches); 0.01 keeps hundreds of crops to minutes on the CPU")
    ap.add_argument("--augment", type=int, default=1, choices=[0, 1, 3, 7],
                    help="extra rotated/flipped copies of each good crop (bricks in different orientations); 0 = off")
    ap.add_argument("--accelerator", default="cpu", choices=["cpu", "xpu"],
                    help="cpu (default) or xpu (Intel GPU; experimental: Patchcore training currently "
                         "triggers an xe driver page fault on this laptop)")
    args = ap.parse_args()

    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine

    good_dir, bad_dir = args.data / "good", args.data / "defective"
    augment_good(good_dir, args.augment)
    n_good = len([p for p in good_dir.glob("*.png") if not p.name.startswith(AUG_PREFIX)])
    n_bad = len(list(bad_dir.glob("*.png"))) if bad_dir.exists() else 0
    if n_good < 5:
        sys.exit(f"need at least 5 images in {good_dir}, found {n_good}")
    print(f"training {args.model} on {n_good} good, {n_bad} defective images from {args.data}")

    def datamodule(batch_size: int):
        return Folder(
            name="bricks",
            root=args.data,
            normal_dir="good",
            abnormal_dir="defective" if n_bad else None,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            seed=0,  # same held-out split every time
        )

    dm = datamodule(1 if args.model == "efficient_ad" else 8)  # EfficientAd requires batch size 1
    model = build_model(args.model, args.coreset)
    work = args.out.parent / f".{args.out.name}-work"
    engine = Engine(max_epochs=args.epochs, default_root_dir=work, **trainer_device(args.accelerator))
    engine.fit(model=model, datamodule=dm)

    # A fixed input size is required: with the default dynamic shape the exported
    # graph has its batch dimension stuck at 2 and does not compile on CPU.
    # A fixed input size is required: with the default dynamic shape the exported
    # graph has its batch dimension stuck at 2 and does not compile on CPU.
    exported = Path(
        engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=work / "export",
            input_size=(args.size, args.size),
            onnx_kwargs={"dynamic_shapes": None},
        )
    )
    targets = [(args.out, "fp")]
    if args.int8:
        targets.append((args.out.with_name(args.out.name + "-int8"), "int8"))
    dm.setup()
    images = heldout_images(dm)
    for out, compression in targets:
        out.mkdir(parents=True, exist_ok=True)
        if compression == "fp":
            for suffix in (".xml", ".bin"):
                shutil.copy(exported.with_suffix(suffix), out / f"model{suffix}")
        else:
            quantize_int8(exported, sorted(good_dir.glob("*.png")), args.size, out / "model.xml")
        scores = by_episode(score_images(out, images))
        threshold, note = pick_threshold(scores)
        meta = {
            "model": args.model,
            "input_size": [args.size, args.size],
            "threshold": threshold,
            "threshold_note": note,
            "compression": compression,
            "train_good": n_good,
            "train_defective": n_bad,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\n{out}: threshold {threshold:.3f} ({note})")
        for path, label, s in sorted(scores, key=lambda r: r[2]):
            wrong = (s >= threshold) != bool(label)
            print(f"  {s:.3f}  {'defective' if label else 'good':9}  {Path(path).name}{'  <-- WRONG' if wrong else ''}")
        print("  (one row per episode: its worst crop)")


if __name__ == "__main__":
    main()
