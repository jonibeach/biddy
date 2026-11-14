"""Deploy Studio's trainer on Modal and register it as a Studio training target.

    ./biddy cloud-trainer            # deploy (or redeploy) and register
    ./biddy cloud-trainer --check    # only ask Studio for the trainer's health

Needs the Modal CLI logged in (`modal token new`) and Studio running.
Steps:
  1. A random path secret is made once and kept in cloud/.trainer_url
     (gitignored) and in the Modal secret `biddy-trainer`.
  2. `modal deploy cloud/modal_trainer.py`.
  3. The URL https://...modal.run/<secret> is registered in Studio as the
     direct training target "modal" (updated in place if it exists).
  4. Studio is asked for the trainer's health (this starts a GPU container,
     which stops again after 20 idle minutes).
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
        out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        return out.stdout + out.stderr
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ""


def deploy() -> str:
    """Deploy and return the full trainer URL (with its secret path)."""
    saved = json.loads(URL_FILE.read_text()) if URL_FILE.exists() else {}
    secret = saved.get("secret") or secrets.token_urlsafe(32)
    # --force replaces the value if the secret already exists; the value is never printed.
    modal("secret", "create", "biddy-trainer", f"TRAINER_PATH_SECRET={secret}", "--force", capture=True)
    out = modal("deploy", "cloud/modal_trainer.py", capture=True)
    print(out[-1500:])
    m = re.search(r"https://\S+?\.modal\.run", out)
    if not m:
        sys.exit("could not find the endpoint URL in modal deploy's output")
    url = f"{m.group(0)}/{secret}"
    URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    URL_FILE.write_text(json.dumps({"base": m.group(0), "secret": secret, "url": url}, indent=2))
    URL_FILE.chmod(0o600)
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
    print("asking Studio for the trainer's health (the first call starts a GPU container; a few minutes)...")
    print(json.dumps(studio("GET", f"/api/remote-trainers/{tid}/health"), indent=2)[:2000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check health of the registered trainer")
    args = ap.parse_args()
    if args.check:
        check()
        return
    url = deploy()
    check(register(url))


if __name__ == "__main__":
    main()
