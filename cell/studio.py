"""Read-only access to Physical AI Studio's database and storage.

Studio keeps robot calibration, serial numbers and camera identities in its
sqlite database, and trained models under its storage folder. Reading them
from there means the cell uses exactly what the demonstrations were recorded
with, and nothing has to be copied by hand.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

STORAGE = Path("~/.local/share/physicalai").expanduser()
DEFAULT_DB = STORAGE / "data" / "physicalai.db"


@dataclass
class RobotRecord:
    name: str
    type: str
    port: str
    serial_number: str
    unit: str
    calibration: dict


@dataclass
class CameraRecord:
    name: str
    driver: str  # "usb_camera" or "realsense"
    fingerprint: dict
    width: int
    height: int
    fps: int


@dataclass
class ModelRecord:
    name: str
    policy: str
    path: Path
    created_at: str

    @property
    def export_dir(self) -> Path:
        return self.path / "exports" / "openvino"


def _connect(db: str | Path) -> sqlite3.Connection:
    db = Path(db).expanduser()
    if not db.exists():
        raise FileNotFoundError(f"Studio database not found at {db}")
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def robot(name: str, db: str | Path = DEFAULT_DB) -> RobotRecord:
    with _connect(db) as c:
        rows = c.execute("select name, type, payload from project_robots where name = ?", (name,)).fetchall()
        if not rows:
            names = [r[0] for r in c.execute("select name from project_robots")]
            raise LookupError(f"no robot named {name!r} in Studio; have {names}")
    if len(rows) > 1:
        raise LookupError(f"{len(rows)} robots named {name!r} in Studio; rename one")
    rname, rtype, payload = rows[0]
    p = json.loads(payload)
    if not p.get("calibration"):
        raise ValueError(f"robot {name!r} has no calibration in Studio; calibrate it there first")
    return RobotRecord(rname, rtype, p.get("connection_string", ""), p.get("serial_number", ""),
                       p.get("unit", "normalized"), p["calibration"])


def cameras(names: list[str], db: str | Path = DEFAULT_DB) -> dict[str, CameraRecord]:
    with _connect(db) as c:
        rows = c.execute("select name, driver, fingerprint, payload from project_cameras").fetchall()
    by_name: dict[str, list] = {}
    for row in rows:
        by_name.setdefault(row[0], []).append(row)
    out = {}
    for name in names:
        found = by_name.get(name, [])
        if len(found) != 1:
            raise LookupError(f"expected one camera named {name!r} in Studio, found {len(found)}; have {sorted(by_name)}")
        _, driver, fingerprint, payload = found[0]
        p = json.loads(payload or "{}")
        out[name] = CameraRecord(name, driver, json.loads(fingerprint or "{}"),
                                 int(p.get("width", 640)), int(p.get("height", 480)), int(p.get("fps", 30)))
    return out


def models(db: str | Path = DEFAULT_DB) -> list[ModelRecord]:
    """Trained models, newest first."""
    with _connect(db) as c:
        rows = c.execute("select name, policy, path, created_at from models order by created_at desc").fetchall()
    return [ModelRecord(n, pol, Path(p).expanduser(), str(t)) for n, pol, p, t in rows]


def resolve_policy_dir(spec: str, db: str | Path = DEFAULT_DB) -> Path:
    """'latest' -> newest Studio model with an OpenVINO export; a Studio model
    name, or 'latest:<policy>' (e.g. latest:act) -> that model's export;
    otherwise a path to the export folder itself or to a Studio model folder.
    """
    path = Path(spec).expanduser()
    if not path.exists():
        if spec == "latest":
            match = lambda m: True  # noqa: E731
        elif spec.startswith("latest:"):
            policy = spec.split(":", 1)[1].lower()
            match = lambda m: (m.policy or "").lower() == policy  # noqa: E731
        else:
            match = lambda m: m.name == spec  # noqa: E731
        for m in models(db):  # newest first
            if match(m) and (m.export_dir / "manifest.json").exists():
                return m.export_dir
        raise FileNotFoundError(f"no Studio model matching {spec!r} with an OpenVINO export yet")
    for candidate in (path, path / "exports" / "openvino"):
        if (candidate / "manifest.json").exists():
            return candidate
    raise FileNotFoundError(f"no manifest.json in {path} or {path / 'exports' / 'openvino'}")


def find_port(record: RobotRecord) -> str:
    """Serial port of the robot, matched by USB serial number when possible.

    /dev/ttyACM numbers swap when the arms are replugged; the serial number doesn't.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        return record.port
    for port in list_ports.comports():
        if record.serial_number and port.serial_number == record.serial_number:
            return port.device
    if record.serial_number:
        raise LookupError(f"{record.name}: no serial port with serial number {record.serial_number}; is it plugged in?")
    return record.port


def main():
    """Print what the cell will use from Studio."""
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config", default="station.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())["studio"]
    r = robot(cfg["follower"], cfg["db"])
    print(f"robot  {r.name}: {find_port(r)} (serial {r.serial_number}, unit {r.unit})")
    for cam in cameras(cfg["cameras"], cfg["db"]).values():
        print(f"camera {cam.name}: {cam.driver} {cam.fingerprint} {cam.width}x{cam.height}@{cam.fps}")
    ms = models(cfg["db"])
    print(f"models {len(ms)} trained" + "".join(f"\n  {m.created_at} {m.name} ({m.policy}) {m.path}" for m in ms))


if __name__ == "__main__":
    main()
