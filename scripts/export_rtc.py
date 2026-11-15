"""Re-export a fine-tuned SmolVLA checkpoint with Real-Time Chunking (RTC) inputs.

    PY=~/physical-ai-studio/application/backend/.venv/bin/python
    $PY scripts/export_rtc.py models/smolvla-ft-grasp55          # -> <model>/exports/openvino-rtc
    $PY scripts/export_rtc.py models/smolvla-ft-grasp55 --check  # also compare against the plain export

RTC is the Robotics AI Suite's way to run a slow flow-matching VLA smoothly
(docs/robotics-ai-suite/.../pi05_with_rtc.md): the next chunk is computed in
the background while the arm keeps moving, and each new chunk is steered
during denoising to continue the unexecuted tail of the previous one, so
chunk seams don't jump. SmolVLA is a flow-matching policy like Pi0.5, and
physicalai's SmolVLA supports it: enabling RTC adds four model inputs
(prev_chunk_left_over, inference_delay, max_guidance_weight,
execution_horizon) that physicalai.runtime.RTCExecution fills in at run time.

Same weights, same training; only the exported graph changes. The plain
export is left alone, so station.yaml can switch with policy.mode: rtc.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cell  # noqa: E402,F401  (sets ZE_ENABLE_ALT_DRIVERS so OpenVINO sees the NPU)

import numpy as np  # noqa: E402


def export(model_dir: Path, out: Path) -> None:
    import torch
    from physicalai.policies.smolvla.policy import SmolVLA

    t0 = time.time()
    policy = SmolVLA.load_from_checkpoint(str(model_dir / "model.ckpt"), map_location="cpu", weights_only=False)
    policy.rtc_enabled = True
    policy.eval()
    if out.exists():
        shutil.rmtree(out)
    print(f"exporting {model_dir.name} with RTC (chunk {policy.config.chunk_size}) -> {out}", flush=True)
    with torch.no_grad():
        policy.export(out, backend="openvino")
    print(f"done in {time.time() - t0:.0f}s")


def check(model_dir: Path, rtc_dir: Path, device: str) -> None:
    """With RTC off (zero guidance), the RTC export must match the plain one; with a
    previous chunk and guidance on, the new chunk's start should follow it."""
    from physicalai.inference import InferenceModel

    from cell.bench import policy_obs

    plain_dir = model_dir / "exports/openvino"
    obs = policy_obs(plain_dir, "pick up the brick")
    rng = np.random.default_rng(0)
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            obs[k] = rng.integers(0, 255, v.shape).astype(v.dtype) if v.dtype == np.uint8 else rng.normal(0, 20, v.shape).astype(v.dtype)

    def run(path, extra):
        m = InferenceModel(path, device=device, INFERENCE_PRECISION_HINT="f32")
        # Fixed noise so both exports start denoising from the same sample.
        np.random.seed(0)
        t = time.perf_counter()
        a = m.predict_action_chunk({**obs, **extra})
        ms = (time.perf_counter() - t) * 1000
        m.close()
        return np.asarray(a).reshape(-1, np.asarray(a).shape[-1]), ms

    chunk = json.loads((rtc_dir / "manifest.json").read_text())
    print("rtc inputs:", [f.get("init_args", f).get("name") for f in chunk["model"]["input_features"]])
    plain, ms_plain = run(plain_dir, {})
    n, d = plain.shape
    off = {"prev_chunk_left_over": np.zeros((1, n, d), np.float32), "inference_delay": np.int64(0),
           "max_guidance_weight": np.float32(1e-8), "execution_horizon": np.int64(1)}
    rtc_off, ms_off = run(rtc_dir, off)
    print(f"plain {ms_plain:.0f} ms, rtc(off) {ms_off:.0f} ms; first-call times include compile")
    print(f"guidance off vs plain: mean |diff| {np.abs(rtc_off - plain).mean():.3f} (noise is sampled per call, so not 0)")
    # Previous chunk = a shifted copy of the plain chunk; with guidance the new chunk's
    # first actions should land closer to it than without.
    prev = np.zeros((1, n, d), np.float32)
    prev[0, : n - 15] = plain[15:]
    on = {"prev_chunk_left_over": prev, "inference_delay": np.int64(3),
          "max_guidance_weight": np.float32(5.0), "execution_horizon": np.int64(15)}
    rtc_on, _ = run(rtc_dir, on)
    head = slice(0, 10)
    print(f"first 10 actions vs previous tail: guided {np.abs(rtc_on[head] - prev[0, head]).mean():.2f}, "
          f"unguided {np.abs(rtc_off[head] - prev[0, head]).mean():.2f} (guided should be smaller)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="folder with model.ckpt, e.g. models/smolvla-ft-grasp55")
    ap.add_argument("--out", help="default: <model>/exports/openvino-rtc")
    ap.add_argument("--check", action="store_true", help="compare against the plain export after exporting")
    ap.add_argument("--check-only", action="store_true", help="skip the export")
    ap.add_argument("--device", default="CPU", help="device for --check")
    args = ap.parse_args()
    model_dir = Path(args.model).resolve()
    out = Path(args.out).resolve() if args.out else model_dir / "exports/openvino-rtc"
    if not args.check_only:
        export(model_dir, out)
    if args.check or args.check_only:
        check(model_dir, out, args.device)


if __name__ == "__main__":
    main()
