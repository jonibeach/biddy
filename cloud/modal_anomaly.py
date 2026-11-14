"""Anomalib (Patchcore) training for every inspection view on a Modal GPU.

    modal run cloud/modal_anomaly.py                          # every view in data/anomaly/
    modal run cloud/modal_anomaly.py --views side,bottom      # some views
    modal run cloud/modal_anomaly.py --args "--coreset 0.02"  # extra anomaly/train.py options
    BIDDY_ANOMALY_GPU=A10G modal run cloud/modal_anomaly.py   # another GPU

Runs anomaly/train.py --accelerator cuda --int8, unchanged, one container per
view in parallel, on the crops in data/anomaly/<view>/ (what anomaly/extract.py
writes). The model folders come back to models/modal/<view>{,-int8}/ on this
laptop, never straight into models/, so a local training run writing
models/<view>* is not overwritten; copy them over when you want to use them:

    cp -r models/modal/* models/

The crops and code are attached to the container at start (Modal only uploads
files it has not seen before), so nothing is stored in Modal between runs.
The held-out report train.py prints is shown here and saved as
models/modal/<view>.log.

Costs: one small GPU per view for a few minutes (L4: about $0.80/h).
"""
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/biddy"
GPU = os.environ.get("BIDDY_ANOMALY_GPU", "L4")
# Which crops to train on (relative to the repo); e.g. BIDDY_ANOMALY_DATA=data/anomaly_masked.
DATA = os.environ.get("BIDDY_ANOMALY_DATA", "data/anomaly")

# Same versions as anomaly/.venv on the laptop, with CUDA instead of XPU torch.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.11.0", "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "anomalib[openvino]==2.7.0", "openvino==2026.4.1", "nncf==3.4.0",
        "opencv-python-headless", "pyyaml",
    )
    # DataLoader workers / multiprocessing need a TMPDIR that can hold Unix sockets.
    .env({"TMPDIR": "/tmp", "PYTHONUNBUFFERED": "1"})
    .add_local_file(ROOT / "anomaly/train.py", f"{REMOTE}/anomaly/train.py")
    .add_local_dir(ROOT / "cell", f"{REMOTE}/cell", ignore=["__pycache__", "**/__pycache__"])
    .add_local_dir(ROOT / DATA, f"{REMOTE}/data/anomaly")
)
app = modal.App("biddy-anomaly")


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, timeout=2 * 3600)
def train_view(view: str, extra: list[str]) -> dict:
    """Train one view; return its model folders as {relative path: bytes} and the log."""
    import subprocess

    out = Path("/tmp/models") / view
    cmd = ["python", "anomaly/train.py", "--data", f"data/anomaly/{view}", "--out", str(out),
           "--int8", "--accelerator", "cuda", *extra]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=REMOTE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        print(f"[{view}] {line}", end="")
        lines.append(line)
    rc = proc.wait()
    seconds = time.time() - t0
    files = {}
    for folder in (out, out.with_name(view + "-int8")):
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file():
                    files[f"{folder.name}/{p.name}"] = p.read_bytes()
    return {"view": view, "returncode": rc, "seconds": seconds, "log": "".join(lines), "files": files}


@app.local_entrypoint()
def main(views: str = "", args: str = "", dest: str = "models/modal"):
    import shlex

    if not views:  # every view with crops (the modal CLI's own Python has no PyYAML for station.yaml)
        views = ",".join(sorted(p.parent.name for p in (ROOT / DATA).glob("*/good")))
    names = [v for v in views.split(",") if v]
    missing = [v for v in names if not (ROOT / DATA / v / "good").is_dir()]
    if missing:
        raise SystemExit(f"no {DATA}/<view>/good for: {', '.join(missing)}")
    extra = shlex.split(args)
    print(f"training {', '.join(names)} on {GPU}")
    t0 = time.time()
    failed = []
    for r in train_view.starmap([(v, extra) for v in names], return_exceptions=True):
        if isinstance(r, Exception):
            print(f"\n=== a view failed: {r!r}")
            failed.append(repr(r))
            continue
        view = r["view"]
        out = ROOT / dest
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{view}.log").write_text(r["log"])
        if r["returncode"] != 0:
            failed.append(view)
            print(f"\n=== {view} FAILED (exit {r['returncode']}) after {r['seconds']:.0f} s; log tail:")
            print("".join(r["log"].splitlines(True)[-40:]))
            continue
        for folder in {k.split("/")[0] for k in r["files"]}:
            # Replace the whole folder so no stale file from an earlier run remains.
            target = out / folder
            if target.exists():
                for p in target.iterdir():
                    p.unlink()
        for rel, data in r["files"].items():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(data)
        report = r["log"][r["log"].find(f"/tmp/models/{view}:"):] if f"/tmp/models/{view}:" in r["log"] else ""
        print(f"\n=== {view}: trained in {r['seconds']:.0f} s on the GPU container -> {out}/{view}{{,-int8}}")
        print(report.replace("/tmp/models/", f"{dest}/"))
    print(f"total wall time {time.time() - t0:.0f} s")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")
