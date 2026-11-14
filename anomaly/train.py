"""Train an Anomalib model on inspection crops and export it for the cell.

Data layout (what anomaly/capture.py writes):

    data/anomaly/good/*.png        bricks without marks, the more the better
    data/anomaly/defective/*.png   marked bricks; only used to test and pick the threshold

Run with the repo's own venv (Anomalib is in the `anomaly` dependency group):

    uv sync --group anomaly
    uv run --group anomaly python anomaly/train.py --data data/anomaly --out models/anomaly

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

MODELS = {
    "patchcore": ("anomalib.models", "Patchcore"),
    "padim": ("anomalib.models", "Padim"),
    "efficient_ad": ("anomalib.models", "EfficientAd"),
}


def build_model(name: str):
    import importlib

    module, cls = MODELS[name]
    return getattr(importlib.import_module(module), cls)()


def heldout_images(datamodule) -> list[tuple[str, int]]:
    """(path, label) of every image the datamodule kept out of training."""
    seen = {}
    for ds in (datamodule.val_data, datamodule.test_data):
        for _, row in ds.samples.iterrows():
            seen[str(row["image_path"])] = int(row["label_index"])
    return sorted(seen.items())


def score_images(model_dir: Path, images: list[tuple[str, int]], device: str = "CPU"):
    m = AnomalyModel(model_dir, device, threshold=0.5)
    scores = []
    for path, label in images:
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        scores.append((path, label, m(rgb).score))
    return scores


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
    ap.add_argument("--accelerator", default="auto", help="Lightning accelerator: auto, cpu, xpu")
    args = ap.parse_args()

    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine

    good_dir, bad_dir = args.data / "good", args.data / "defective"
    n_good = len(list(good_dir.glob("*.png")))
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
    model = build_model(args.model)
    work = args.out.parent / f".{args.out.name}-work"
    engine = Engine(max_epochs=args.epochs, default_root_dir=work, accelerator=args.accelerator)
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
        scores = score_images(out, images)
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


if __name__ == "__main__":
    main()
