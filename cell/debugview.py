"""Live debug view of the cell: what the gripper is going for, the phase, the plan.

    ./biddy run --debug            # then open http://localhost:8765 (the browser pane works)

One picture, refreshed ~5x a second, streamed as MJPEG and saved as
runs/<time>/debug.mp4:

  top camera      bricks found against the empty table (green boxes), the
                  gripper found by its pink colour (magenta), and the brick
                  nearest to it, i.e. the one it is going for (yellow, "TARGET")
  wrist camera    a box where the jaws close: whatever is in it gets grasped
  inspection cam  as is, with the current inspection pose when scripted
  panel           phase, attempt, time in phase; gripper measured / commanded /
                  planned (the policy's queued chunk: when it intends to close);
                  each joint now vs where the queued chunk ends (the policy's
                  near-term target)

The top-camera target is an estimate from colour (no camera-to-robot
calibration): good enough to see "it aims at the gap next to the brick".
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

W, H = 320, 240           # each camera tile
PANEL_H = 230
JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
JAW_BOX = (440, 110, 170, 150)   # wrist camera (640x480): just past the jaw tips (from a 17:52 frame)


def _pink(hsv: np.ndarray) -> np.ndarray:
    h, s = hsv[..., 0], hsv[..., 1]
    return ((h >= 140) | (h <= 8)) & (s > 70)


def _card_mask(hsv: np.ndarray) -> np.ndarray:
    """The white card under this camera's light: bluish (hue 85-130), bright, mildly saturated."""
    h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 85) & (h <= 130) & (sat < 90) & (v > 185)


def _card_region(rgb: np.ndarray):
    """(card mask, filled card outline) for the largest card-coloured area, or None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.morphologyEx(_card_mask(hsv).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 5000:
        return None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [cv2.convexHull(c)], -1, 255, -1)
    return mask, filled, hsv


def find_card(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of the card the bricks lie on."""
    r = _card_region(rgb)
    return None if r is None else tuple(int(v) for v in cv2.boundingRect(r[1]))


def find_bricks(rgb: np.ndarray, empty: np.ndarray | None = None, edges: bool = False) -> list[tuple[int, int, int, int]]:
    """Things lying on the card: holes in the card's colour inside its outline, minus the jaws.

    `empty` is unused (kept for callers).
    """
    r = _card_region(rgb)
    if r is None:
        return []
    mask, filled, hsv = r
    inner = cv2.erode(filled, np.ones((31, 31), np.uint8))  # off the card edge, its tape and the tags beside it
    holes = (inner > 0) & (mask == 0) & ~_pink(hsv) & (hsv[..., 2] > 90)
    holes = cv2.morphologyEx(holes.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(holes)
    # Bricks seen from above are 20-55 px; a taller strip is the card's own edge or a fold.
    found = [tuple(int(v) for v in stats[k][:4]) for k in range(1, n)
             if 150 < stats[k][4] < 5000 and max(stats[k][2], stats[k][3]) <= 60]
    if edges:
        for b in _edge_bricks(rgb, filled):
            if not any(_overlap(b, f) for f in found):
                found.append(b)
    return found


def _overlap(a, b) -> bool:
    return not (a[0] + a[2] < b[0] or b[0] + b[2] < a[0] or a[1] + a[3] < b[1] or b[1] + b[3] < a[1])


def _edge_bricks(rgb: np.ndarray, card_filled: np.ndarray) -> list[tuple[int, int, int, int]]:
    """White bricks on the white card: sharp edges (outline, studs, print). A shadow's edge
    is soft and does not pass the high Canny thresholds. Use with the arm away (at home):
    the arm itself is full of edges."""
    inner = cv2.erode(card_filled, np.ones((31, 31), np.uint8))  # off the card edge, tape and tags
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    e = cv2.Canny(g, 30, 80)  # low enough for a plain white brick; used at home, with no claw shadow on the card
    e[inner == 0] = 0
    e = cv2.morphologyEx(cv2.dilate(e, np.ones((7, 7), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(e)
    return [tuple(int(v) for v in stats[k][:4]) for k in range(1, n)
            if 300 < stats[k][4] < 6000 and max(stats[k][2], stats[k][3]) <= 60]


def find_gripper(rgb: np.ndarray) -> tuple[int, int] | None:
    """Centre of the jaws: the largest strongly pink blob that is not the (bigger) pink bowl."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (_pink(hsv) & (hsv[..., 1] > 110)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    ok = [k for k in range(1, n) if 150 < stats[k][4] < 8000]
    if not ok:
        return None
    card = find_card(rgb)
    if card is not None:  # the pink bowl lies beside the card; the jaws work over it
        cx, cy = card[0] + card[2] / 2, card[1] + card[3] / 2
        k = min(ok, key=lambda k: (cents[k][0] - cx) ** 2 + (cents[k][1] - cy) ** 2)
    else:
        k = max(ok, key=lambda k: stats[k][4])
    return int(cents[k][0]), int(cents[k][1])


class DebugView:
    def __init__(self, cameras: dict, out_dir: Path, port: int | None = None, fps: float = 5.0,
                 latest: Path | None = None):
        """port: serve the stream here (None: don't); latest: also write each frame to this
        file, for the always-on page (scripts/debug_server.py)."""
        self.cameras, self.out_dir, self.fps = cameras, Path(out_dir), fps
        self.lock = threading.Lock()
        self.state: np.ndarray | None = None
        self.cmd: np.ndarray | None = None
        self.plan: np.ndarray | None = None      # queued actions, (n, 6)
        self.phase, self.detail, self.phase_t0 = "starting", "", time.monotonic()
        self.empty_top: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.video = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._server = None
        self.latest = latest
        if port is not None:
            try:
                self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
                threading.Thread(target=self._server.serve_forever, daemon=True).start()
                print(f"debug view: http://localhost:{port}", flush=True)
            except OSError as e:
                print(f"debug view: no web stream ({e}); still writing {self.out_dir}/debug.mp4", flush=True)
        self._thread.start()

    # --- inputs ---------------------------------------------------------------

    def set_phase(self, phase: str, detail: str = ""):
        with self.lock:
            if phase != self.phase:
                self.phase_t0 = time.monotonic()
            self.phase, self.detail = phase, detail

    def set_empty_top(self, rgb: np.ndarray | None):
        with self.lock:
            self.empty_top = None if rgb is None else rgb.copy()

    def update(self, state=None, cmd=None, plan=None):
        with self.lock:
            if state is not None:
                self.state = np.asarray(state, np.float32).copy()
            if cmd is not None:
                self.cmd = np.asarray(cmd, np.float32).copy()
            self.plan = None if plan is None else np.asarray(plan, np.float32).copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self.video is not None:
            self.video.release()
        if self._server is not None:
            self._server.shutdown()

    # --- drawing --------------------------------------------------------------

    def _frame(self, name: str) -> np.ndarray | None:
        cam = self.cameras.get(name)
        if cam is None:
            return None
        try:
            return cam.read_latest().data
        except Exception:  # noqa: BLE001 - a camera hiccup must not break the view
            return None

    def _tile(self, rgb: np.ndarray | None, label: str) -> np.ndarray:
        if rgb is None:
            img = np.zeros((H, W, 3), np.uint8)
        else:
            img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (W, H))
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def compose(self) -> np.ndarray:
        with self.lock:
            state, cmd, plan = self.state, self.cmd, self.plan
            phase, detail, t_phase = self.phase, self.detail, time.monotonic() - self.phase_t0
            empty = self.empty_top
        top, wrist, insp = self._frame("top"), self._frame("wrist"), self._frame("inspection")

        t = self._tile(top, "top")
        if top is not None:
            sx, sy = W / top.shape[1], H / top.shape[0]
            card = find_card(top)
            if card is not None:
                x, y, w, h = card
                cv2.rectangle(t, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), (120, 120, 120), 1)
            bricks = find_bricks(top)
            grip = find_gripper(top)
            for (x, y, w, h) in bricks:
                cv2.rectangle(t, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), (0, 200, 0), 1)
            if grip is not None:
                gx, gy = int(grip[0] * sx), int(grip[1] * sy)
                cv2.drawMarker(t, (gx, gy), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
                if bricks:
                    x, y, w, h = min(bricks, key=lambda b: (b[0] + b[2] / 2 - grip[0]) ** 2 + (b[1] + b[3] / 2 - grip[1]) ** 2)
                    c = (int((x + w / 2) * sx), int((y + h / 2) * sy))
                    cv2.rectangle(t, (int(x * sx) - 2, int(y * sy) - 2), (int((x + w) * sx) + 2, int((y + h) * sy) + 2),
                                  (0, 255, 255), 2)
                    cv2.line(t, (gx, gy), c, (0, 255, 255), 1)
                    d = np.hypot(c[0] - gx, c[1] - gy) / sx
                    cv2.putText(t, f"TARGET {d:.0f}px", (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 255), 1, cv2.LINE_AA)

        wv = self._tile(wrist, "wrist")
        if wrist is not None:
            sx, sy = W / wrist.shape[1], H / wrist.shape[0]
            x, y, w, h = JAW_BOX
            cv2.rectangle(wv, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), (0, 255, 255), 1)
            cv2.putText(wv, "jaws close here", (int(x * sx), int(y * sy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1, cv2.LINE_AA)
        iv = self._tile(insp, "inspection")
        row = np.hstack([t, wv, iv])

        panel = np.full((PANEL_H, row.shape[1], 3), 24, np.uint8)
        put = lambda txt, x, y, col=(230, 230, 230), sc=0.55: cv2.putText(  # noqa: E731
            panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        put(f"PHASE  {phase.upper()}   {detail}", 10, 24, (0, 255, 255), 0.65)
        put(f"{t_phase:5.1f} s in phase", 10, 48)
        # gripper: measured / commanded / planned
        g_state = float(state[5]) if state is not None else float("nan")
        g_cmd = float(cmd[5]) if cmd is not None else float("nan")
        put(f"gripper  measured {g_state:5.1f}   commanded {g_cmd:5.1f}", 10, 74)
        bx, bw = 10, 400
        cv2.rectangle(panel, (bx, 84), (bx + bw, 98), (80, 80, 80), 1)
        if np.isfinite(g_state):
            cv2.rectangle(panel, (bx, 84), (bx + int(bw * np.clip(g_state, 0, 100) / 100), 98), (0, 200, 0), -1)
        if np.isfinite(g_cmd):
            xc = bx + int(bw * np.clip(g_cmd, 0, 100) / 100)
            cv2.line(panel, (xc, 80), (xc, 102), (0, 165, 255), 2)
        if plan is not None and len(plan):
            gp = plan[:, 5]
            pts = [(bx + int(i * bw / max(1, len(gp) - 1)), 150 - int(np.clip(v, 0, 100) * 0.4)) for i, v in enumerate(gp)]
            cv2.polylines(panel, [np.array(pts, np.int32)], False, (0, 165, 255), 1)
            close = next((i for i, v in enumerate(gp) if v < 12), None)
            put("planned gripper (next %d actions)" % len(gp), bx, 160, (0, 165, 255), 0.45)
            put("closes in %s" % (f"{close / 30:.1f} s" if close is not None else "-  (not in this chunk)"),
                bx + 260, 160, (0, 165, 255), 0.45)
        # joints: now vs end of the queued chunk
        x0 = 470
        put("joint     now   plan-end", x0, 74, (200, 200, 200), 0.5)
        for j, name in enumerate(JOINTS):
            now = float(state[j]) if state is not None else float("nan")
            end = float(plan[-1, j]) if plan is not None and len(plan) else float("nan")
            col = (0, 165, 255) if np.isfinite(end) and abs(end - now) > 5 else (200, 200, 200)
            put(f"{name:6} {now:7.1f}  {end:7.1f}", x0, 98 + 20 * j, col, 0.5)
        put("top: grey = card, green = coloured bricks (white ones not found), magenta = gripper, yellow = nearest brick (estimate)", 10, PANEL_H - 12,
            (150, 150, 150), 0.42)
        return np.vstack([row, panel])

    def _loop(self):
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                img = self.compose()
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.jpeg = buf.tobytes()
                    if self.latest is not None:  # atomic: the page never reads half a file
                        tmp = self.latest.with_suffix(".tmp")
                        tmp.write_bytes(self.jpeg)
                        tmp.replace(self.latest)
                if self.video is None:
                    self.out_dir.mkdir(parents=True, exist_ok=True)
                    self.video = cv2.VideoWriter(str(self.out_dir / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 self.fps, (img.shape[1], img.shape[0]))
                self.video.write(img)
            except Exception as e:  # noqa: BLE001
                print(f"debug view: {type(e).__name__}: {e}", flush=True)
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))

    def _handler(self):
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while not view._stop.is_set():
                            if view.jpeg:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + view.jpeg + b"\r\n")
                            time.sleep(1.0 / view.fps)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                body = (b"<!doctype html><title>cell debug</title><body style='margin:0;background:#111'>"
                        b"<img src='/stream' style='width:100%;max-width:960px;display:block;margin:auto'></body>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler
