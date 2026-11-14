"""Run SmolVLA locally through OpenVINO: build, optimize and benchmark exports.

    PY=~/physical-ai-studio/application/backend/.venv/bin/python
    cd ~/biddy

    # (a) base checkpoint (lerobot/smolvla_base) exported with OUR features:
    #     3 cameras top/inspection/wrist 480x640, 6-D state/action, our stats
    $PY scripts/smolvla_local.py build                      # -> models/smolvla-base/exports/openvino

    # (b) weight compression of ANY export dir (base or the fine-tuned one);
    #     writes a sibling folder <export>-int8 (or --out) and leaves the input alone
    $PY scripts/smolvla_local.py optimize --export models/smolvla-base/exports/openvino --mode int8_sym
    $PY scripts/smolvla_local.py optimize --export ~/.local/share/physicalai/models/<id>/exports/openvino

    # (c) latency + accuracy vs the FP32 CPU reference, JSON into runs/
    $PY scripts/smolvla_local.py bench --export models/smolvla-base/exports/openvino \
        models/smolvla-base/exports/openvino-int8_sym --devices GPU NPU CPU

Accuracy is measured on real frames from the Studio dataset with a fixed start
noise, against the first export run on CPU with INFERENCE_PRECISION_HINT=f32.
Errors are in action units (normalized joint degrees) and as a fraction of the
action standard deviation.

NNCF is not in Studio's venv; `optimize` looks for it in models/.pylib, which
is filled once (no change to the Studio venv) with:

    uv pip install --python $PY --target models/.pylib --no-deps nncf jsonschema \
        jsonschema-specifications referencing rpds-py attrs natsort pydot pyparsing \
        tabulate scikit-learn joblib threadpoolctl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cell  # noqa: E402,F401  (sets ZE_ENABLE_ALT_DRIVERS so OpenVINO sees the NPU)

import numpy as np  # noqa: E402

DATASET = Path.home() / ".local/share/physicalai/datasets/7760da20-1077-402d-82d7-a1e85bac7a02"
DEFAULT_OUT = REPO / "models/smolvla-base"
PYLIB = REPO / "models/.pylib"
CACHE = REPO / "models/.ov_cache"
TASK = "put the brick in the good bowl"


# --------------------------------------------------------------------------- build
def dataset_module(root: Path):
    from physicalai.data import LeRobotDataModule

    dm = LeRobotDataModule(repo_id="snapshot", root=str(root), train_batch_size=1, num_workers=0)
    dm.setup("fit")
    return dm


def cmd_build(args):
    """Base weights + our dataset's features/stats -> OpenVINO export, no training.

    SmolVLA(pretrained) loads the base config/weights; replacing its
    preprocessor stats with our dataset's is exactly what Studio's fit() does in
    setup() before the first step. Export then traces our 3 cameras and 6-D
    state, giving the same manifest layout as a Studio-trained model.
    """
    from physicalai.policies import get_policy

    t0 = time.time()
    stats = dataset_module(Path(args.dataset)).train_dataset.stats
    policy = get_policy("smolvla", pretrained_name_or_path=args.base)
    policy._update_preprocessor_stats(stats)  # noqa: SLF001 - what SmolVLA.setup() does on fit
    policy.eval()
    out = Path(args.out) / "exports/openvino"
    if out.exists():
        shutil.rmtree(out)
    print(f"exporting {args.base} with features {[f.name for f in policy.inputs_schema]} -> {out}")
    policy.export(out, backend="openvino")
    print(f"done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------ optimize
def policy_xml(export_dir: Path) -> Path:
    xmls = [p for p in export_dir.glob("*.xml") if "tokenizer" not in p.name]
    if len(xmls) != 1:
        raise SystemExit(f"expected one policy .xml in {export_dir}, found {[p.name for p in xmls]}")
    return xmls[0]


def fix_graph(model) -> dict:
    """bf16 compute -> f32, Bucketize i64 -> i32. Weights stay as stored.

    SmolVLA trains and exports with dtype=bfloat16, so the IR computes most
    MatMuls in bf16. This laptop's CPU has no native bf16 (emulated, slow) and
    the NPU compiler rejects bf16 ops and an i64 Bucketize. The GPU runs its
    own f16 either way. Only element types change, never the math's structure.
    """
    import openvino as ov
    from openvino import opset13 as ops

    bf16, n = ov.Type.bf16, {"bf16_constants": 0, "bf16_converts": 0, "bucketize": 0}
    for o in model.get_ordered_ops():
        t = o.get_type_name()
        if t == "Constant" and o.get_output_element_type(0) == bf16:
            targets = list(o.output(0).get_target_inputs())
            cv = ops.convert(o.output(0), "f32")
            for ti in targets:
                ti.replace_source_output(cv.output(0))
            n["bf16_constants"] += 1
        elif t == "Convert" and o.get_output_element_type(0) == bf16:
            o.output(0).replace(ops.convert(o.input_value(0), "f32").output(0))
            n["bf16_converts"] += 1
        elif t == "Bucketize" and o.get_output_element_type(0) == ov.Type.i64:
            nb = ops.bucketize(o.input_value(0), o.input_value(1), output_type="i32",
                               with_right_bound=o.get_attributes()["with_right_bound"])
            o.output(0).replace(ops.convert(nb, "i64").output(0))
            n["bucketize"] += 1
    model.validate_nodes_and_infer_types()
    return n


def cmd_optimize(args):
    if PYLIB.exists():
        sys.path.insert(0, str(PYLIB))
    import openvino as ov

    src = Path(args.export).expanduser().resolve()
    name = "-".join([*(["f32"] if args.fix else []), *([args.mode] if args.mode != "none" else [])]) or "copy"
    dst = Path(args.out) if args.out else src.with_name(f"{src.name}-{name}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    xml = policy_xml(dst)
    t0 = time.time()
    core = ov.Core()
    model = core.read_model(str(xml))
    info: dict = {"source": str(src)}
    if args.fix:
        info["fix_graph"] = fix_graph(model)
    compressed = model
    if args.mode != "none":
        import nncf

        kw = {"mode": getattr(nncf.CompressWeightsMode, args.mode.upper())}
        if args.mode.startswith("int4"):
            kw.update(group_size=args.group_size, ratio=args.ratio)
        info.update({k: str(v) for k, v in kw.items()})
        compressed = nncf.compress_weights(model, **kw)
    tmp = dst / "_tmp"
    tmp.mkdir()
    ov.save_model(compressed, str(tmp / xml.name), compress_to_fp16=False)
    del compressed, model, core
    (tmp / xml.name).replace(xml)
    (tmp / xml.with_suffix(".bin").name).replace(xml.with_suffix(".bin"))
    tmp.rmdir()
    (dst / "optimization.json").write_text(json.dumps(info, indent=2))
    mb = lambda p: p.with_suffix(".bin").stat().st_size / 1e6  # noqa: E731
    print(f"{name}: {mb(policy_xml(src)):.0f} MB -> {mb(xml):.0f} MB in {time.time() - t0:.0f}s: {dst}")


# --------------------------------------------------------------------------- bench
def real_observations(export_dir: Path, n: int, dataset: Path) -> tuple[list[dict], list[np.ndarray]]:
    """n frames spread over the dataset in the cell's input format, plus their demo action chunks."""
    manifest = json.loads((export_dir / "manifest.json").read_text())
    feats = [f.get("init_args", f) for f in manifest["model"]["input_features"]]
    noise_shape = next((tuple(f["shape"]) for f in feats if f["name"] == "noise"), (50, 32))
    ds = dataset_module(dataset).train_dataset
    raw = ds._lerobot_dataset.hf_dataset  # noqa: SLF001 - action/episode columns without video decode
    episodes = np.asarray(raw["episode_index"])
    rng = np.random.default_rng(0)
    obs, demo = [], []
    # Spread over the dataset, offset so frames are mid-episode rather than at the home pose.
    for i in (np.linspace(0, len(ds) - 1, n + 1)[:-1] + len(ds) / (2 * n)).astype(int):
        item = ds[int(i)]
        item = item.to_dict() if hasattr(item, "to_dict") else item
        o = {"task": [item.get("task") or TASK], "noise": rng.standard_normal((1, *noise_shape), dtype=np.float32)}
        for f in feats:
            if f.get("ftype") == "VISUAL":
                img = np.asarray(item[f["name"]], np.float32)  # (3,H,W) in [0,1]
                o[f["name"]] = (np.clip(img, 0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)[None]
            elif f.get("ftype") == "STATE":
                o["state"] = np.asarray(item["state"], np.float32).reshape(1, -1)
        obs.append(o)
        j = np.arange(i, min(i + noise_shape[0], len(raw)))
        j = j[episodes[j] == episodes[i]]
        demo.append(np.asarray(raw[j]["action"], np.float32))
    return obs, demo


def stage_timer(model):
    """Wrap the preprocessors and the adapter so one call's time splits into stages."""
    times: dict[str, float] = {}

    def wrap(name, fn):
        def inner(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            times[name] = times.get(name, 0.0) + (time.perf_counter() - t) * 1000
            return r
        return inner

    model.preprocessors = [wrap(type(p).__name__, p) for p in model.preprocessors]
    model.adapter.predict = wrap("policy_graph", model.adapter.predict)
    return times


def run_config(export_dir: Path, device: str, cfg: dict, obs: list[dict], runs: int, ref: list | None, act_std, demo):
    from physicalai.inference import InferenceModel

    row = {"export": str(export_dir), "device": device, "config": cfg}
    try:
        t = time.perf_counter()
        m = InferenceModel(export_dir, device=device, **cfg)
        row["load_s"] = round(time.perf_counter() - t, 2)
        t = time.perf_counter()
        first = m.predict_action_chunk(obs[0])
        row["first_call_ms"] = round((time.perf_counter() - t) * 1000, 1)
        outs = [first] + [m.predict_action_chunk(o) for o in obs[1:]]
        row["chunk"] = list(first.shape)
        row["finite"] = bool(all(np.all(np.isfinite(o)) for o in outs))
        if ref is not None:
            d = np.stack([np.abs(o - r) for o, r in zip(outs, ref)])
            row["max_abs_err"] = round(float(d.max()), 4)
            row["mean_abs_err"] = round(float(d.mean()), 4)
            row["mean_err_over_std"] = round(float((d / act_std).mean()), 4)
        row["mean_abs_err_vs_demo"] = round(float(np.mean([np.abs(o[: len(d)] - d).mean() for o, d in zip(outs, demo)])), 3)
        stages = stage_timer(m)
        lat = []
        for i in range(runs):
            t = time.perf_counter()
            m.predict_action_chunk(obs[i % len(obs)])
            lat.append((time.perf_counter() - t) * 1000)
        lat.sort()
        row["median_ms"] = round(statistics.median(lat), 1)
        row["p90_ms"] = round(lat[int(len(lat) * 0.9)], 1)
        row["stages_ms"] = {k: round(v / runs, 1) for k, v in stages.items()}
        m.close()
        return row, outs
    except Exception as e:  # noqa: BLE001 - a device that cannot run it is a result too
        row["error"] = f"{type(e).__name__}: {e}"[:2000]
        return row, None


def cmd_bench(args):
    import openvino as ov

    devs = ov.Core().available_devices
    exports = [Path(e).expanduser().resolve() for e in args.export]
    obs, demo = real_observations(exports[0], args.frames, Path(args.dataset))
    stats = json.loads((exports[0] / "manifest.json").read_text())
    act_std = 1.0
    try:  # action std from the denormalize postprocessor, for a scale-free error
        for p in stats["model"]["postprocessors"]:
            s = p.get("init_args", p).get("stats", {}).get("action")
            if s:
                act_std = np.asarray(s["std"], np.float32)
    except Exception:  # noqa: BLE001
        pass
    print(f"OpenVINO {ov.__version__}, devices {devs}; {len(obs)} real frames, {args.runs} timed runs")
    out = REPO / "runs" / f"smolvla-bench-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():  # after every row: a GPU plugin abort (e.g. int4) kills the process, not just the row
        out.write_text(json.dumps({"openvino": ov.__version__, "devices": devs, "frames": len(obs),
                                   "runs": args.runs, "results": rows}, indent=2))

    print("reference: first export on CPU f32 ...")
    ref_row, ref = run_config(exports[0], "CPU", {"INFERENCE_PRECISION_HINT": "f32"}, obs, args.runs, None, act_std, demo)
    ref_row["reference"] = True
    rows = [ref_row]
    if ref is None:
        raise SystemExit(f"reference failed: {ref_row['error']}")

    configs = {
        "GPU": [{}, {"INFERENCE_PRECISION_HINT": "f32"}, {"PERFORMANCE_HINT": "THROUGHPUT"}],
        "NPU": [{}],
        "CPU": [{}, {"PERFORMANCE_HINT": "THROUGHPUT"}],
    }
    if args.quick:
        configs = {"GPU": [{}], "NPU": [{}], "CPU": [{}]}
    for export_dir in exports:
        for dev in args.devices:
            if dev not in devs:
                rows.append({"export": str(export_dir), "device": dev, "error": f"{dev} not in {devs}"})
                continue
            for cfg in configs.get(dev, [{}]):
                cfg = {"PERFORMANCE_HINT": "LATENCY", **cfg}
                if args.cache and dev != "CPU":  # CPU compiles in seconds; its blobs are ~1 GB each
                    cfg["CACHE_DIR"] = str(CACHE)
                print(f"{export_dir.name} {dev} {cfg} ...", flush=True)
                row, _ = run_config(export_dir, dev, cfg, obs, args.runs, ref, act_std, demo)
                rows.append(row)
                save()
                print("   ", {k: v for k, v in row.items() if k not in ("export", "config")}, flush=True)

    print(f"\n{'export':28} {'device':5} {'config':34} {'load':>6} {'median':>8} {'p90':>8} {'err/std':>8}")
    for r in rows:
        cfg = ",".join(f"{k.split('_')[0][:4]}={v}" for k, v in r.get("config", {}).items() if k != "CACHE_DIR")
        name = Path(r["export"]).name if Path(r["export"]).name != "openvino" else Path(r["export"]).parents[1].name
        if "error" in r:
            print(f"{name[:28]:28} {r['device']:5} {cfg[:34]:34} ERROR {r['error'].splitlines()[0][:100]}")
        else:
            print(f"{name[:28]:28} {r['device']:5} {cfg[:34]:34} {r['load_s']:5.1f}s {r['median_ms']:7.0f}ms "
                  f"{r['p90_ms']:7.0f}ms {r.get('mean_err_over_std', 0):8.4f}" + ("" if r["finite"] else "  NaN"))
    save()
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="export lerobot/smolvla_base with our dataset's features")
    b.add_argument("--base", default="lerobot/smolvla_base")
    b.add_argument("--dataset", default=str(DATASET))
    b.add_argument("--out", default=str(DEFAULT_OUT))
    o = sub.add_parser("optimize", help="NNCF weight compression of any export dir")
    o.add_argument("--export", required=True)
    o.add_argument("--mode", default="int8_sym", help="none, int8_sym, int8_asym, int4_sym, int4_asym")
    o.add_argument("--no-fix", dest="fix", action="store_false", help="keep bf16 compute (see fix_graph)")
    o.add_argument("--group-size", type=int, default=16, help="int4 only; must divide 720 and 960 (SmolVLA widths)")
    o.add_argument("--ratio", type=float, default=1.0, help="int4 only: share of layers in int4 (rest int8)")
    o.add_argument("--out")
    be = sub.add_parser("bench", help="latency and accuracy per device; JSON to runs/")
    be.add_argument("--export", nargs="+", required=True, help="first one is the FP32 CPU reference")
    be.add_argument("--devices", nargs="+", default=["GPU", "NPU", "CPU"],
                    help="NPU: an unfixed export fails in seconds (bf16/Bucketize); a fix_graph'd one "
                         "compiled for >45 min without finishing, so leave NPU out for those")
    be.add_argument("--dataset", default=str(DATASET))
    be.add_argument("--frames", type=int, default=8)
    be.add_argument("--runs", type=int, default=20)
    be.add_argument("--quick", action="store_true", help="only the default config per device")
    be.add_argument("--no-cache", dest="cache", action="store_false", help="no OpenVINO CACHE_DIR")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    {"build": cmd_build, "optimize": cmd_optimize, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
