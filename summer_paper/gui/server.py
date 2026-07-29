"""
Backend for the interactive dragonfly trajectory GUI.

A dependency-free (stdlib only) HTTP server that:

  * serves the single-file frontend at ``GET /``;
  * runs one :class:`~dragonfly_sim.Simulator` continuously in a background
    thread (real-time paced against the wall clock x ``time_scale``);
  * streams render state to the browser at ~60 Hz over Server-Sent Events
    (``GET /stream``);
  * accepts commands as JSON POSTs: ``/path`` (a freehand polyline to follow),
    ``/target`` (a moving target to pursue, or ``{"clear": true}``),
    ``/params`` (live morphology / gains / follow speed) and ``/reset``.

Why SSE + POST rather than websockets: it needs no third-party package, works on
the project ``dragonpy`` env as-is, and the draw -> follow -> hover -> draw loop
is naturally one continuous server->client stream plus discrete client->server
commands.

Run:  PYENV_VERSION=dragonpy pyenv exec python -m summer_paper.gui.server
then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dragonfly_sim import Simulator, Config  # noqa: E402

INDEX_HTML = HERE / "index.html"
HOST, PORT = "127.0.0.1", 8765

# Morphology bounds (Table 1) the server clamps incoming params to.
BOUNDS = {
    "Lw": (0.5, 0.95), "A_over_m": (0.15, 1.2), "omega_star": (6.0, 22.0),
    "Kd": (0.1, 12.0), "v_follow": (0.05, 3.0),
    "taper": (0.0, 2.0), "time_scale": (0.1, 3.0),
}

# Pursuit-target speed cap (BL per sqrt(L/g)); absurd aim drags are clamped.
TARGET_VMAX = 5.0

# Where committed drawings are persisted for the report scripts.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class SimRunner:
    """Owns the simulator and drives it in real time on a daemon thread."""

    def __init__(self):
        self.sim = Simulator()
        self.lock = self.sim.lock
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._clients = 0                       # live SSE connections
        self._clients_lock = threading.Lock()

    def start(self):
        self.thread.start()

    # -- SSE client accounting (gates the sim so it idles when unwatched) -----

    def add_client(self):
        with self._clients_lock:
            self._clients += 1

    def remove_client(self):
        with self._clients_lock:
            self._clients = max(0, self._clients - 1)

    def has_clients(self):
        with self._clients_lock:
            return self._clients > 0

    def _loop(self):
        substeps = 80                                   # integration steps/beat
        wall_prev = time.perf_counter()
        sim_debt = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            elapsed = now - wall_prev
            wall_prev = now
            # Idle when no browser is watching: don't integrate, wake rarely, and
            # drop any accrued debt so reconnecting resumes from the frozen state
            # (self.t hasn't advanced, so an in-progress path resumes seamlessly).
            if not self.has_clients():
                sim_debt = 0.0
                time.sleep(0.1)
                continue
            try:
                blew_up = False
                with self.lock:
                    scale = self.sim.cfg.time_scale
                    # recompute dt if omega changed
                    dt = (2.0 * np.pi / self.sim.cfg.omega_star) / substeps
                    sim_debt += elapsed * scale
                    # advance, capped so a hiccup can't explode the step budget
                    max_steps = substeps * 3
                    steps = min(int(sim_debt / dt), max_steps)
                    for _ in range(steps):
                        self.sim.step(dt)
                    sim_debt -= steps * dt
                    if steps == max_steps:
                        sim_debt = 0.0                   # drop backlog, stay live
                    if not np.isfinite(self.sim.s).all():
                        self.sim.reset()                 # blew up (unstable gains)
                        sim_debt = 0.0
                        blew_up = True
                if blew_up:
                    print("[sim] non-finite state (unstable gains?); auto-reset",
                          file=sys.stderr)
            except Exception as exc:                     # never let the loop die
                print(f"[sim] loop error: {exc!r}; auto-reset", file=sys.stderr)
                try:
                    with self.lock:
                        self.sim.reset()
                except Exception:
                    pass
                sim_debt = 0.0
                time.sleep(0.05)
            time.sleep(0.002)

    # -- command surface (thread-safe) --------------------------------------

    def snapshot(self):
        with self.lock:
            return self.sim.snapshot()

    def set_path(self, points):
        with self.lock:
            anchor = (float(self.sim.s[0]), float(self.sim.s[2]))
            path = self.sim.set_path(points)
            v_follow, taper = self.sim.cfg.v_follow, self.sim.cfg.taper
        if path is not None:
            self._dump_path(points, anchor, path, v_follow, taper)
        return None if path is None else path.tolist()

    @staticmethod
    def _dump_path(points, anchor, path, v_follow, taper):
        """Persist the last committed drawing (raw + smoothed) so the report's
        prescribed-trajectory figure (scripts/trajectory_gains.py) can replay
        it. Overwritten on every committed path; best-effort (a failed write
        never disturbs the sim)."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            out = {"points": [[float(x), float(z)] for x, z in points],
                   "anchor": [anchor[0], anchor[1]],
                   "smoothed": [[float(x), float(z)] for x, z in path],
                   "v_follow": float(v_follow), "taper": float(taper)}
            with open(DATA_DIR / "drawn_path.json", "w") as f:
                json.dump(out, f)
        except OSError as e:
            print(f"drawn-path dump failed: {e}", file=sys.stderr)

    def set_target(self, d):
        try:
            x, z = float(d["x"]), float(d["z"])
            vx, vz = float(d.get("vx", 0.0)), float(d.get("vz", 0.0))
        except (KeyError, TypeError, ValueError):
            return False
        if not all(np.isfinite(val) for val in (x, z, vx, vz)):
            return False
        spd = float(np.hypot(vx, vz))
        if spd > TARGET_VMAX:
            vx, vz = vx * TARGET_VMAX / spd, vz * TARGET_VMAX / spd
        with self.lock:
            self.sim.set_target(x, z, vx, vz)
        return True

    def clear_target(self):
        with self.lock:
            self.sim.clear_target()

    def set_params(self, d):
        with self.lock:
            for k, (lo, hi) in BOUNDS.items():
                if k in d:
                    try:
                        setattr(self.sim.cfg, k, float(np.clip(float(d[k]), lo, hi)))
                    except (TypeError, ValueError):
                        pass
        return self.config()

    def reset(self):
        with self.lock:
            self.sim.reset()

    def config(self):
        c = self.sim.cfg
        return {"Lw": c.Lw, "A_over_m": c.A_over_m, "omega_star": c.omega_star,
                "Kd": c.Kd, "v_follow": c.v_follow,
                "taper": c.taper, "time_scale": c.time_scale}


RUNNER = SimRunner()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"          # simple per-request close; SSE streams until done

    def log_message(self, *args):
        pass                               # quiet

    # -- helpers -------------------------------------------------------------

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode())
        except json.JSONDecodeError:
            return {}

    # -- routes --------------------------------------------------------------

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                body = INDEX_HTML.read_bytes()
            except FileNotFoundError:
                self.send_error(404, "index.html not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/config":
            with RUNNER.lock:
                self._send_json(RUNNER.config())
        elif self.path == "/stream":
            self._stream()
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        RUNNER.add_client()                 # un-pauses the sim
        try:
            while True:
                snap = RUNNER.snapshot()
                payload = "data: " + json.dumps(snap) + "\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
                time.sleep(1.0 / 60.0)
        except (BrokenPipeError, ConnectionResetError):
            pass                            # client navigated away
        finally:
            RUNNER.remove_client()          # re-pauses when the last one leaves

    def do_POST(self):
        if self.path == "/path":
            d = self._read_json()
            pts = [(float(p[0]), float(p[1])) for p in d.get("points", [])]
            path = RUNNER.set_path(pts)
            self._send_json({"ok": path is not None, "path": path})
        elif self.path == "/target":
            d = self._read_json()
            if d.get("clear"):
                RUNNER.clear_target()
                self._send_json({"ok": True})
            else:
                self._send_json({"ok": RUNNER.set_target(d)})
        elif self.path == "/params":
            self._send_json(RUNNER.set_params(self._read_json()))
        elif self.path == "/reset":
            RUNNER.reset()
            self._send_json({"ok": True})
        else:
            self.send_error(404)


def main():
    RUNNER.start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"dragonfly GUI serving at {url}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
