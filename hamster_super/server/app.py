"""HTTP server: runs the control loop and serves a live dashboard.

Endpoints
    GET  /                 dashboard
    GET  /api/state        JSON snapshot (pose, belief walls, path, rays, ...)
    GET  /api/camera.jpg   latest camera frame (with wall mask overlay)
    POST /api/start        start driving
    POST /api/stop         emergency stop
    POST /api/reset        reset belief/odometry to the start pose   body: {"goal": [x, y]} optional
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

DASHBOARD = Path(__file__).with_name("dashboard.html")


def make_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, DASHBOARD.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(runtime.snapshot())
            elif path == "/api/camera.jpg":
                jpg = camera_jpeg(runtime)
                if jpg is None:
                    self._send(404, b"no frame", "text/plain")
                else:
                    self._send(200, jpg, "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            path = self.path.split("?")[0]
            if path == "/api/start":
                runtime.start()
            elif path == "/api/stop":
                runtime.stop()
            elif path == "/api/reset":
                runtime.stop()
                if hasattr(runtime.hw, "reset_robot"):
                    runtime.hw.reset_robot()
                runtime.reset(goal=body.get("goal"))
                runtime.status = "idle"
            else:
                return self._send(404, b"not found", "text/plain")
            self._json({"ok": True, "status": runtime.status})

    return Handler


def camera_jpeg(runtime):
    frame = runtime.frame
    if frame is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    img = frame.copy()
    mask = runtime.perception.last_mask
    if mask is not None and mask.shape == img.shape[:2]:
        img[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 0, 69])).astype(np.uint8)
    img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else None


def serve(runtime, host="0.0.0.0", port=8000):
    httpd = ThreadingHTTPServer((host, port), make_handler(runtime))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
