"""Always-on debug page for the cell: http://localhost:8765

    ./biddy debug-server            # leave it running in a terminal

Shows the cell's live debug view (cameras, the brick it is heading for,
phase, gripper plan; see cell/debugview.py) whenever a run is going, and the
last frame marked IDLE when none is. It never opens the cameras or the arm
itself (they belong to the cell while it runs).

The toggle on the page switches the debug view for the next runs on or off
(file runs/.debug_off); a run started with --debug shows it either way.
"""
from __future__ import annotations

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
LATEST = RUNS / "debug_latest.jpg"
OFF = RUNS / ".debug_off"
STALE_S = 3.0

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>cell debug</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:8px 12px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid #555;background:#222;color:#ddd;cursor:pointer}
button.on{background:#1d5e2b;border-color:#2e8b43}
img{display:block;width:100%;max-width:1100px;margin:0 auto}</style></head><body>
<header><b>cell debug</b><button id=t onclick="fetch('/toggle',{method:'POST'}).then(r=>r.text()).then(s)">...</button>
<span id=st></span></header><img src="/stream">
<script>function s(v){const b=document.getElementById('t');b.textContent='debug view: '+v;b.className=v=='on'?'on':''}
fetch('/state').then(r=>r.text()).then(s);
setInterval(()=>fetch('/status').then(r=>r.text()).then(t=>document.getElementById('st').textContent=t),1000)</script>
</body></html>"""


def idle_frame() -> bytes:
    """The last debug frame, dimmed and marked IDLE; a placeholder if there is none."""
    img = cv2.imread(str(LATEST)) if LATEST.exists() else None
    if img is None:
        img = np.full((470, 960, 3), 24, np.uint8)
        msg = "no run yet - start one with ./biddy run"
    else:
        img = (img * 0.35).astype(np.uint8)
        msg = "IDLE - no run active (last frame %s)" % time.strftime("%H:%M:%S", time.localtime(LATEST.stat().st_mtime))
    cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    if OFF.exists():
        cv2.putText(img, "debug view switched off for the next runs", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def live() -> bool:
    return LATEST.exists() and time.time() - LATEST.stat().st_mtime < STALE_S


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _text(self, body: str, ctype: str = "text/plain"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path == "/toggle":
            RUNS.mkdir(exist_ok=True)
            if OFF.exists():
                OFF.unlink()
            else:
                OFF.touch()
            return self._text("off" if OFF.exists() else "on")
        self.send_error(404)

    def do_GET(self):
        if self.path == "/state":
            return self._text("off" if OFF.exists() else "on")
        if self.path == "/status":
            return self._text("LIVE run" if live() else "idle")
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_idle = 0.0
            idle = None
            try:
                while True:
                    if live():
                        try:
                            jpg = LATEST.read_bytes()
                        except OSError:
                            jpg = None
                        time.sleep(0.2)
                    else:
                        if idle is None or time.time() - last_idle > 2.0:
                            idle, last_idle = idle_frame(), time.time()
                        jpg = idle
                        time.sleep(0.5)
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        return self._text(PAGE, "text/html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"cell debug page: http://localhost:{args.port}  (Ctrl+C stops)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
