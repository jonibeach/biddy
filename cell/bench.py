"""Time the cell's models on every Intel device: CPU, GPU and NPU.

    PY=~/physical-ai-studio/application/backend/.venv/bin/python
    $PY -m cell.bench                         # models from station.yaml
    $PY -m cell.bench --policy latest:act --anomaly models/anomaly models/anomaly-int8

Prints a table (median and p90 latency per call) and saves it as JSON under
runs/, for picking devices in station.yaml and for the demo write-up. Inputs
are zeros shaped from each model's manifest, so no hardware is needed.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from cell import studio
from cell.anomaly import AnomalyModel
from cell.intel import devices


def _time(fn, warmup: int, runs: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.9)]


def policy_obs(export_dir: Path, task: str) -> dict:
    manifest = json.loads((export_dir / "manifest.json").read_text())
    obs = {}
    for f in manifest["model"]["input_features"]:
        a = f.get("init_args", f)
        if a.get("ftype") == "VISUAL":
            c, h, w = a["shape"]
            obs[a["name"]] = np.zeros((1, h, w, c), np.uint8)
        elif a.get("ftype") == "STATE":
            obs[a["name"]] = np.zeros((1, *a["shape"]), np.float32)
    obs["task"] = [task]
    return obs


def bench_policy(export_dir: Path, task: str, devs: list[str], runs: int) -> list[dict]:
    from physicalai.inference import InferenceModel

    obs = policy_obs(export_dir, task)
    rows = []
    for dev in devs:
        for precision in (["f16", "f32"] if dev == "GPU" else ["default"]):
            kwargs = {"INFERENCE_PRECISION_HINT": "f32"} if precision == "f32" else {}
            row = {"model": f"policy {export_dir.parent.parent.name if export_dir.name == 'openvino' else export_dir.name}",
                   "device": dev, "precision": precision}
            try:
                m = InferenceModel(export_dir, device=dev, **kwargs)
                chunk = m.predict_action_chunk(obs)
                row["nan"] = bool(not np.all(np.isfinite(chunk)))
                row["chunk"] = list(chunk.shape)
                row["median_ms"], row["p90_ms"] = _time(lambda: m.predict_action_chunk(obs), 3, runs)
                m.close()
            except Exception as e:  # noqa: BLE001 - a device that cannot run it is a result too
                row["error"] = str(e).splitlines()[0][:120]
            rows.append(row)
    return rows


def bench_anomaly(model_dir: Path, devs: list[str], runs: int) -> list[dict]:
    rows = []
    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    for dev in devs:
        row = {"model": f"anomaly {model_dir.name}", "device": dev, "precision": "model"}
        try:
            m = AnomalyModel(model_dir, dev)
            row["median_ms"], row["p90_ms"] = _time(lambda: m(img), 5, runs)
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e).splitlines()[0][:120]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--policy", nargs="*", help="policy specs (default: policy.model and policy.drop_model)")
    ap.add_argument("--anomaly", nargs="*", help="anomaly model dirs (default: inspection.model)")
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    devs = devices()
    print(f"OpenVINO devices: {devs}")

    rows = []
    specs = args.policy if args.policy is not None else [s for s in (cfg["policy"]["model"], cfg["policy"].get("drop_model")) if s]
    for spec in specs:
        try:
            export_dir = studio.resolve_policy_dir(spec, cfg["studio"]["db"])
        except FileNotFoundError as e:
            print(f"skip policy {spec}: {e}")
            continue
        rows += bench_policy(export_dir, cfg["policy"]["tasks"]["good"], devs, args.runs)
    for d in args.anomaly if args.anomaly is not None else [cfg["inspection"]["model"]]:
        if not (Path(d) / "model.xml").exists():
            print(f"skip anomaly {d}: no model.xml")
            continue
        rows += bench_anomaly(Path(d), devs, args.runs)

    print(f"\n{'model':40} {'device':6} {'precision':9} {'median':>9} {'p90':>9}")
    for r in rows:
        if "error" in r:
            res = f"cannot run: {r['error']}"
        else:
            res = f"{r['median_ms']:8.1f}ms {r['p90_ms']:8.1f}ms" + ("  NaN OUTPUT" if r.get("nan") else "")
        print(f"{r['model'][:40]:40} {r['device']:6} {r['precision']:9} {res}")
    out = Path(cfg["runs_dir"]) / f"bench-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"devices": devs, "results": rows}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
