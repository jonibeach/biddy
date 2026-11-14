"""Score one inspection image with an exported Anomalib model on OpenVINO.

Only needs numpy, OpenCV and OpenVINO, so it runs in the Studio venv next to
the policy. The model folder is what `anomaly/train.py` writes:

    <model_dir>/model.xml, model.bin   OpenVINO IR, static [1, 3, H, W] input
    <model_dir>/meta.json              input size, threshold, crop it was trained on

Anomalib puts its ImageNet normalization inside the exported graph, so the
input is RGB float in [0, 1], resized here to the model's fixed size. Its `pred_score` is already normalized so that 0.5 is the
threshold it picked on validation; meta.json can override that.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Verdict:
    score: float  # normalized anomaly score, higher = more defective
    defective: bool
    latency_ms: float
    anomaly_map: np.ndarray | None = None  # HxW float, same size as model input


def crop(image: np.ndarray, roi: list[int] | None) -> np.ndarray:
    """Cut [x, y, w, h] out of an image; None returns the image unchanged."""
    if not roi:
        return image
    x, y, w, h = roi
    H, W = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi} is outside the {W}x{H} image")
    return image[y0:y1, x0:x1]


class AnomalyModel:
    def __init__(self, model_dir: str | Path, device: str = "CPU", threshold: float | None = None):
        import openvino as ov

        model_dir = Path(model_dir)
        xml = model_dir / "model.xml"
        if not xml.exists():
            raise FileNotFoundError(f"no model.xml in {model_dir}; run anomaly/train.py first")
        meta_path = model_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.threshold = threshold if threshold is not None else float(self.meta.get("threshold", 0.5))
        core = ov.Core()
        model = core.read_model(xml)
        shape = model.inputs[0].partial_shape
        if shape.is_dynamic:
            raise ValueError(f"{xml} has a dynamic input {shape}; export it with a fixed input_size")
        _, _, self.height, self.width = (d.get_length() for d in shape)
        self.device = device
        self.compiled = core.compile_model(model, device)
        self._outputs = {o.get_any_name(): o for o in self.compiled.outputs}

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    def __call__(self, rgb: np.ndarray, with_map: bool = False) -> Verdict:
        """Score an RGB uint8 image that is already cropped to the trained ROI."""
        t0 = time.perf_counter()
        result = self.compiled(self.preprocess(rgb))
        latency = (time.perf_counter() - t0) * 1000
        score = float(np.asarray(result[self._outputs["pred_score"]]).reshape(-1)[0])
        amap = None
        if with_map and "anomaly_map" in self._outputs:
            amap = np.asarray(result[self._outputs["anomaly_map"]])[0, 0]
        return Verdict(score, score >= self.threshold, latency, amap)


def overlay(rgb: np.ndarray, verdict: Verdict) -> np.ndarray:
    """BGR image with the anomaly heat map and verdict drawn on, for a preview window."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if verdict.anomaly_map is not None:
        amap = cv2.resize(verdict.anomaly_map, (bgr.shape[1], bgr.shape[0]))
        heat = cv2.applyColorMap(np.clip(amap * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        bgr = cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)
    label = f"{'DEFECT' if verdict.defective else 'GOOD'} {verdict.score:.2f}"
    color = (0, 0, 255) if verdict.defective else (0, 200, 0)
    cv2.putText(bgr, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return bgr


def main():
    ap = argparse.ArgumentParser(description="Score images or time an exported anomaly model.")
    ap.add_argument("model_dir")
    ap.add_argument("images", nargs="*", help="image files to score (already cropped)")
    ap.add_argument("--device", default="CPU", help="OpenVINO device: CPU, GPU, NPU")
    ap.add_argument("--bench", action="store_true", help="time 50 runs on every available device")
    args = ap.parse_args()

    if args.bench:
        import openvino as ov

        dummy = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        for dev in ov.Core().available_devices:
            try:
                m = AnomalyModel(args.model_dir, dev)
            except Exception as e:  # noqa: BLE001 - report and keep going
                print(f"{dev}: cannot compile ({str(e).splitlines()[0]})")
                continue
            for _ in range(5):
                m(dummy)
            times = sorted(m(dummy).latency_ms for _ in range(50))
            print(f"{dev}: median {times[25]:.1f} ms, p90 {times[45]:.1f} ms")
        return

    m = AnomalyModel(args.model_dir, args.device)
    for path in args.images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"{path}: cannot read")
            continue
        v = m(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        print(f"{path}: score {v.score:.3f} -> {'DEFECT' if v.defective else 'good'} ({v.latency_ms:.1f} ms)")


if __name__ == "__main__":
    main()
