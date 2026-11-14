"""Train a policy in Physical AI Studio from the command line, and wait for it.

Uses Studio's own REST API (the same call its Train button makes), so the
model lands in Studio with its OpenVINO export and `policy.model: latest` in
station.yaml picks it up. Studio must be running (scripts/studio.sh).

    PY=~/physical-ai-studio/application/backend/.venv/bin/python
    $PY -m cell.train_policy --policy smolvla --name smolvla-v1 --epochs 20
    $PY -m cell.train_policy --policy act --name act-pick --trainer <remote trainer name>
    $PY -m cell.train_policy --list            # datasets, trainers, jobs

--trainer picks a remote (cloud GPU) trainer registered in Studio under
Settings > Training Targets; without it Studio trains locally (Intel GPU/CPU).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from cell import dataset as dataset_mod
from cell import studio


def api(url: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
            return json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Studio API {method} {url} -> {e.code}: {e.read().decode()[:2000]}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach Studio at {url} ({e.reason}); start it with scripts/studio.sh") from None


def _rows(db, sql, args=()):
    with sqlite3.connect(f"file:{Path(db).expanduser()}?mode=ro", uri=True) as c:
        return c.execute(sql, args).fetchall()


def find_dataset(spec: str, db) -> tuple[str, str, str]:
    """(dataset_id, project_id, name) for 'latest', a Studio dataset name, or a dataset folder."""
    if spec == "latest":
        rows = _rows(db, "select id, project_id, name, path from datasets order by created_at desc")
    else:
        rows = _rows(db, "select id, project_id, name, path from datasets where name = ? or path = ?",
                     (spec, str(Path(spec).expanduser())))
    for did, pid, name, path in rows:
        if (Path(path) / "meta" / "info.json").exists():
            return did, pid, name
    if rows:
        raise SystemExit(f"dataset {rows[0][2]!r} has no files yet; end the recording session in Studio first")
    raise SystemExit(f"no Studio dataset {spec!r}")


def find_trainer(name: str, db) -> str:
    rows = _rows(db, "select id, name from remote_trainers")
    for tid, tname in rows:
        if tname == name or tid == name:
            return tid
    raise SystemExit(f"no remote trainer {name!r} in Studio; have {[r[1] for r in rows]} "
                     "(add one under Settings > Training Targets)")


def list_all(base: str, db):
    print("datasets:")
    for did, name, path, created in _rows(db, "select id, name, path, created_at from datasets order by created_at desc"):
        ready = (Path(path) / "meta" / "info.json").exists()
        print(f"  {created}  {name}  {did}  {'ready' if ready else 'NOT WRITTEN YET (recording open?)'}")
    print("remote trainers:")
    for tid, name, mode in _rows(db, "select id, name, connection_mode from remote_trainers"):
        print(f"  {name}  {tid}  ({mode})")
    print("jobs:")
    for j in api(f"{base}/api/jobs") or []:
        print(f"  {j.get('id')}  {j.get('status')}  {j.get('progress')}%  {j.get('message') or ''}")
    print("models:")
    for m in studio.models(db):
        print(f"  {m.created_at}  {m.name} ({m.policy})  export: {m.export_dir if m.export_dir.exists() else 'none'}")


def wait(base: str, job_id: str, poll_s: float = 15.0) -> dict:
    last = None
    while True:
        job = api(f"{base}/api/jobs/{job_id}")
        line = f"{job.get('status')} {job.get('progress')}% {job.get('message') or ''}"
        if line != last:
            print(time.strftime("%H:%M:%S"), line, flush=True)
            last = line
        if job.get("status") in ("completed", "failed", "canceled"):
            return job
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="station.yaml")
    ap.add_argument("--list", action="store_true", help="show datasets, trainers, jobs and models, then exit")
    ap.add_argument("--policy", default="smolvla", help="act, smolvla, pi05, ...")
    ap.add_argument("--name", help="model name in Studio (default: <policy>-<time>)")
    ap.add_argument("--dataset", default="latest", help="'latest', a Studio dataset name, or its folder")
    ap.add_argument("--epochs", type=int, default=None, help="default: Studio's default")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--trainer", help="remote trainer name or id; default trains locally")
    ap.add_argument("--device", help="local only: xpu or cpu (default: Studio picks)")
    ap.add_argument("--augment", action="store_true", help="image augmentation (helps new brick positions)")
    ap.add_argument("--no-wait", action="store_true", help="submit and return")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    db = cfg["studio"]["db"]
    base = cfg["studio"].get("url", "http://localhost:7860").rstrip("/")
    if args.list:
        list_all(base, db)
        return

    dataset_id, project_id, ds_name = find_dataset(args.dataset, db)
    # Catch obvious recording problems before spending GPU time.
    eps = dataset_mod.Dataset(dataset_mod.resolve(ds_name if args.dataset == "latest" else args.dataset, db)).episodes()
    print(f"dataset {ds_name}: {len(eps)} episodes, tasks {sorted({e.task for e in eps})}")

    name = args.name or f"{args.policy}-{time.strftime('%m%d-%H%M')}"
    payload = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "policy": args.policy,
        "model_name": name,
        "batch_size": args.batch_size,
        "augment_images": args.augment,
        "export_backends": ["openvino"],
    }
    if args.epochs:
        payload["max_epochs"] = args.epochs
    if args.trainer:
        payload["training_target"] = "remote"
        payload["remote_trainer_id"] = find_trainer(args.trainer, db)
    else:
        payload["training_target"] = "local"
        if args.device:
            payload["device"] = {"type": args.device, "index": 0 if args.device == "xpu" else None}
    job = api(f"{base}/api/jobs:train", "POST", payload)
    print(f"submitted job {job['id']}: {args.policy} '{name}' on {args.trainer or 'local'}")
    if args.no_wait:
        return
    job = wait(base, job["id"])
    if job.get("status") != "completed":
        sys.exit(f"training {job.get('status')}: {job.get('message')}")
    for m in studio.models(db):
        if m.name == name:
            print(f"model ready: {m.export_dir}")
            break


if __name__ == "__main__":
    main()
