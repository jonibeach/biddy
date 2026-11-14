"""Deploy Studio's trainer on Modal and register it as a Studio training target.

    ./biddy cloud-trainer            # deploy (or redeploy) and register
    ./biddy cloud-trainer --check    # only ask Studio for the trainer's health

Needs the Modal CLI logged in (`modal token new`) and Studio running.
Steps:
  1. A random path secret is made once and kept in cloud/.trainer_url
     (gitignored) and in the Modal secret `biddy-trainer`.
  2. `modal run --detach cloud/modal_trainer.py` starts the trainer on a GPU
     behind a Modal tunnel (unless one is already running), which stops by
     itself after 30 idle minutes.
  3. The URL https://<tunnel>/<secret> is registered in Studio as the direct
     training target "modal" (updated in place if it exists).
  4. Waits until Studio reports the trainer healthy.
Then: ./biddy train-policy --trainer modal --policy smolvla --name <name>
"""
from __future__ import annotations

import argparse
import json
import re
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL_FILE = ROOT / "cloud" / ".trainer_url"
STUDIO = "http://localhost:7860"
NAME = "modal"


def studio(method: str, path: str, body: dict | None = None, timeout: float = 600):
    req = urllib.request.Request(f"{STUDIO}{path}", method=method, headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode()
            return json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        sys.exit(f"Studio {method} {path} -> {e.code}: {e.read().decode()[:1000]}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach Studio at {STUDIO} ({e.reason}); start it with scripts/studio.sh")


def modal(*args: str, capture: bool = False) -> str:
    cmd = ["modal", *args]
    if capture:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if out.returncode:
            sys.exit(f"{' '.join(cmd[:3])} failed:\n{(out.stdout + out.stderr)[-2000:]}")
        return out.stdout + out.stderr
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ""


def running_url() -> str | None:
    """The saved trainer URL if a trainer is answering there right now."""
    if not URL_FILE.exists():
        return None
    url = json.loads(URL_FILE.read_text()).get("url")
    if not url:
        return None
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=15) as r:
            return url if json.loads(r.read()).get("status") == "healthy" else None
    except Exception:  # noqa: BLE001 - not running
        return None


def start() -> str:
    """Start the trainer on Modal (or reuse a running one); return its full URL with the secret path."""
    if url := running_url():
        print("a trainer is already running; reusing it")
        return url
    saved = json.loads(URL_FILE.read_text()) if URL_FILE.exists() else {}
    secret = saved.get("secret") or secrets.token_urlsafe(32)
    # From a private temp file, not argv (other users can read process arguments);
    # --force replaces the value if the secret exists. The value is never printed.
    import os
    import tempfile

    fd, env_file = tempfile.mkstemp(suffix=".env")
    try:
        os.write(fd, f"TRAINER_PATH_SECRET={secret}\n".encode())
        os.close(fd)
        modal("secret", "create", "biddy-trainer", "--from-dotenv", env_file, "--force", capture=True)
    finally:
        os.unlink(env_file)
    # The old web-endpoint deployment (cut uploads at ~150 s) must not linger.
    subprocess.run(["modal", "app", "stop", "biddy-trainer", "--yes"], cwd=ROOT, capture_output=True)
    out = modal("run", "--detach", "cloud/modal_trainer.py", capture=True)
    m = re.search(r"TRAINER_TUNNEL=(https://\S+)", out)
    if not m:
        sys.exit(f"no tunnel URL in modal run's output:\n{out[-1500:]}")
    url = f"{m.group(1).rstrip('/')}/{secret}"
    URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    URL_FILE.write_text(json.dumps({"base": m.group(1), "secret": secret, "url": url}, indent=2))
    URL_FILE.chmod(0o600)
    print("trainer started on Modal")
    return url


def register(url: str) -> str:
    """Create or update Studio's direct training target NAME; returns its id."""
    body = {"name": NAME, "connection": {"connection_mode": "direct", "url": url}}
    existing = [t for t in studio("GET", "/api/remote-trainers") or [] if t.get("name") == NAME]
    if existing:
        tid = existing[0]["id"]
        studio("PATCH", f"/api/remote-trainers/{tid}", body)
        print(f"updated Studio training target '{NAME}'")
    else:
        tid = studio("POST", "/api/remote-trainers", body)["id"]
        print(f"added Studio training target '{NAME}'")
    return tid


def check(tid: str | None = None):
    if tid is None:
        found = [t for t in studio("GET", "/api/remote-trainers") or [] if t.get("name") == NAME]
        if not found:
            sys.exit(f"no Studio training target '{NAME}'; run without --check first")
        tid = found[0]["id"]
    import time

    for _ in range(20):  # Studio waits only a few seconds per check; a cold trainer needs longer
        health = studio("GET", f"/api/remote-trainers/{tid}/health")
        if health.get("status") == "healthy":
            break
        time.sleep(15)
    print(json.dumps(health, indent=2)[:2000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check health of the registered trainer")
    args = ap.parse_args()
    if args.check:
        check()
        return
    url = start()
    check(register(url))


if __name__ == "__main__":
    main()
