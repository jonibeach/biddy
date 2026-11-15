"""Merge Studio datasets into one LeRobot folder for cloud/modal_finetune.py.

    python scripts/merge_datasets.py --out data/finetune/all2 pick-inspect-sort long-bricks anomaly-extra pick-spread-good
    python scripts/merge_datasets.py --out data/finetune/all2 --all     # every Studio dataset with episodes

Datasets are given by Studio name or id. Episodes listed in
data/finetune/exclude.txt ("<dataset id> <episode index>" per line, # comments)
are left out: those datasets are first copied without them to a temp folder.
Studio's own datasets are only read, never changed.
Run with Studio's backend Python (it has lerobot):
    ~/physical-ai-studio/application/backend/.venv/bin/python scripts/merge_datasets.py ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDIO = Path.home() / ".local/share/physicalai"
EXCLUDE = ROOT / "data/finetune/exclude.txt"


def studio_datasets() -> dict[str, str]:
    """name -> id, from Studio's database."""
    db = sqlite3.connect(STUDIO / "data/physicalai.db")
    return {name: did for did, name in db.execute("select id, name from datasets")}


def excluded() -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    if EXCLUDE.exists():
        for line in EXCLUDE.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                did, ep = line.split()
                out.setdefault(did, set()).add(int(ep))
    return out


def episodes(did: str) -> int:
    info = STUDIO / "datasets" / did / "meta/info.json"
    return json.loads(info.read_text())["total_episodes"] if info.exists() else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="Studio dataset names or ids")
    ap.add_argument("--all", action="store_true", help="every Studio dataset with episodes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.datasets.dataset_tools import aggregate_datasets, delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = studio_datasets()
    ids = {v: k for k, v in names.items()}
    wanted = [d for d in names.values() if episodes(d) > 0] if args.all else [names.get(d, d) for d in args.datasets]
    unknown = [d for d in wanted if d not in ids]
    if unknown or not wanted:
        raise SystemExit(f"unknown datasets: {unknown}; Studio has {sorted(names)}")
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick another --out or remove it")
    skip = excluded()
    tmp = Path(tempfile.mkdtemp(prefix="merge_", dir=ROOT / "data/finetune"))
    try:
        roots = []
        for did in wanted:
            src = STUDIO / "datasets" / did
            drop = sorted(e for e in skip.get(did, ()) if e < episodes(did))
            if drop:
                ds = LeRobotDataset(repo_id=f"local/{did}", root=src)
                dst = tmp / did
                delete_episodes(ds, drop, output_dir=dst, repo_id=f"local/{did}")
                roots.append(dst)
                print(f"{ids[did]}: {episodes(did)} episodes, leaving out {drop}")
            else:
                roots.append(src)
                print(f"{ids[did]}: {episodes(did)} episodes")
        aggregate_datasets(repo_ids=[f"local/{r.name}" for r in roots], aggr_repo_id=f"local/{out.name}",
                           roots=roots, aggr_root=out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    info = json.loads((out / "meta/info.json").read_text())
    print(f"merged {info['total_episodes']} episodes, {info['total_frames']} frames -> {out}")


if __name__ == "__main__":
    main()
