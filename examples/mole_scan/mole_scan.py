#!/usr/bin/env python3
"""
mole_scan.py - run the mole detector live on a SunFounder PiDog.

  python3 mole_scan.py                      # Pi camera, web view on :8000
  python3 mole_scan.py --pidog              # also sit + sweep head + LEDs
  python3 mole_scan.py --source 0           # USB webcam (laptop testing)
  python3 mole_scan.py --source arm.jpg     # still image / video file

Open http://<pidog-ip>:8000 for the annotated stream.
Snapshots (raw + annotated + JSON) land in ./scans/<date>/.

Stop any other program using the camera first (e.g. the SunFounder app /
vilib examples), otherwise Picamera2 can't open it.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from mole_detector import DetectorConfig, annotate, detect


# ---------------------------------------------------------------- camera ----
class Source:
    def __init__(self, spec: str, size=(1280, 960)):
        self.still = None
        self.cam = self.cap = None
        if spec == "picam":
            from picamera2 import Picamera2
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(main={"size": size, "format": "RGB888"})
            self.cam.configure(cfg)
            self.cam.start()
            try:  # Camera Module 3 has autofocus; OV5647 (stock PiDog) ignores this
                self.cam.set_controls({"AfMode": 2})
            except Exception:
                pass
            time.sleep(1.0)
        elif spec.isdigit():
            self.cap = cv2.VideoCapture(int(spec))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        else:
            img = cv2.imread(spec)
            if img is not None:
                self.still = img
            else:
                self.cap = cv2.VideoCapture(spec)

    def read(self):
        if self.still is not None:
            time.sleep(0.2)
            return self.still.copy()
        if self.cam is not None:
            return self.cam.capture_array()  # "RGB888" is BGR byte order = OpenCV-ready
        ok, frame = self.cap.read()
        if not ok and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:  # loop video files
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cam is not None:
            self.cam.stop()
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------- pidog ----
class DogBody:
    """Thin, fail-soft wrapper over SunFounder's `pidog` library."""

    # (yaw, roll, pitch) head poses to sweep through
    POSES = [(-30, 0, -15), (0, 0, -15), (30, 0, -15),
             (30, 0, 5), (0, 0, 5), (-30, 0, 5)]

    def __init__(self, sweep=True, dwell=2.5, bark=False):
        from pidog import Pidog
        self.dog = Pidog()
        self.sweep, self.dwell, self.bark = sweep, dwell, bark
        self.paused = threading.Event()   # set => hold current pose
        self.stop = threading.Event()
        self._last_alert = 0.0
        self.dog.do_action("sit", speed=50)
        self.dog.wait_all_done()
        self.leds("breath", "cyan")

    def leds(self, style, color, bps=1.0):
        try:
            self.dog.rgb_strip.set_mode(style=style, color=color, bps=bps, brightness=0.8)
        except Exception:
            pass

    def head(self, yaw, roll, pitch):
        try:
            # -35 = ActionFlow.SIT_HEAD_PITCH; keeps the head level while sitting
            self.dog.head_move([[yaw, roll, pitch]], pitch_comp=-35, immediately=True, speed=60)
            self.dog.wait_head_done()
        except Exception as e:
            print("head_move failed:", e)

    def run_sweep(self):
        i = 0
        while not self.stop.is_set():
            if self.sweep and not self.paused.is_set():
                self.head(*self.POSES[i % len(self.POSES)])
                i += 1
            self.stop.wait(self.dwell)

    def found(self, n):
        if n:
            self.paused.set()          # hold still so you get a sharp snapshot
            self.leds("boom", "pink", bps=2)
            if self.bark and time.time() - self._last_alert > 10:
                self._last_alert = time.time()
                try:
                    self.dog.speak("single_bark_1", 60)
                except Exception:
                    pass
        else:
            if self.paused.is_set():
                self.leds("breath", "cyan")
            self.paused.clear()

    def close(self):
        self.stop.set()
        try:
            self.head(0, 0, 0)
            self.dog.do_action("lie", speed=40)
            self.dog.wait_all_done()
            self.dog.close()
        except Exception:
            pass


# ------------------------------------------------------------- web view ----
class Shared:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = b""
        self.spots = []
        self.snap_request = False


PAGE = b"""<!doctype html><meta name=viewport content="width=device-width">
<title>PiDog mole scan</title>
<style>body{font:14px system-ui;margin:12px;background:#111;color:#eee}
img{max-width:100%;border-radius:8px}button{padding:8px 14px;margin:8px 0}
pre{background:#222;padding:8px;border-radius:6px;max-height:40vh;overflow:auto}</style>
<h3>PiDog mole scan <small style="color:#999">tracking aid, not a diagnosis</small></h3>
<img src="/stream.mjpg"><br>
<button onclick="fetch('/snapshot',{method:'POST'})">Save snapshot</button>
<pre id=j></pre>
<script>setInterval(async()=>{j.textContent=JSON.stringify(
 await (await fetch('/spots.json')).json(),null,1)},1000)</script>"""


def make_handler(shared: Shared):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path == "/snapshot":
                shared.snap_request = True
                self.send_response(204)
                self.end_headers()

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/spots.json":
                body = json.dumps(shared.spots).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        with shared.cond:
                            shared.cond.wait(timeout=5)
                            frame = shared.jpeg
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: %d\r\n\r\n" % len(frame))
                        self.wfile.write(frame + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)
    return H


# ------------------------------------------------------------------ main ----
def save_snapshot(root: Path, raw, ann, spots):
    stamp = datetime.now()
    d = root / stamp.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    base = d / stamp.strftime("%H%M%S")
    cv2.imwrite(f"{base}_raw.jpg", raw)
    cv2.imwrite(f"{base}_annotated.jpg", ann)
    Path(f"{base}.json").write_text(json.dumps(
        {"time": stamp.isoformat(), "spots": spots}, indent=2))
    print(f"saved {base}  ({len(spots)} spots)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="picam", help="picam | webcam index | image/video path")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pidog", action="store_true", help="drive the PiDog (sit, sweep head, LEDs)")
    ap.add_argument("--no-sweep", action="store_true", help="with --pidog: keep head still")
    ap.add_argument("--bark", action="store_true", help="with --pidog: bark when a spot is found")
    ap.add_argument("--px-per-mm", type=float, help="calibration, to report sizes in mm")
    ap.add_argument("--contrast", type=int, default=16, help="darkness threshold (lower = more sensitive)")
    ap.add_argument("--stable-frames", type=int, default=6, help="frames a spot must persist before auto-save")
    ap.add_argument("--cooldown", type=float, default=15, help="seconds between auto-saves")
    ap.add_argument("--no-autosave", action="store_true")
    ap.add_argument("--show-skin", action="store_true", help="tint the detected skin area")
    ap.add_argument("--scans", default="scans")
    args = ap.parse_args()

    cfg = DetectorConfig(px_per_mm=args.px_per_mm, contrast_thresh=args.contrast)
    src = Source(args.source)
    shared = Shared()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(shared))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"web view: http://<this-pi>:{args.port}")

    body = None
    if args.pidog:
        body = DogBody(sweep=not args.no_sweep, bark=args.bark)
        threading.Thread(target=body.run_sweep, daemon=True).start()

    streak, last_save = 0, 0.0
    try:
        while True:
            frame = src.read()
            if frame is None:
                time.sleep(0.05)
                continue
            # Detect on a downscaled copy for speed on the Pi, then rescale.
            scale = 640 / frame.shape[1] if frame.shape[1] > 640 else 1.0
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            if args.px_per_mm:
                cfg.px_per_mm = args.px_per_mm * scale
            spots, skin = detect(small, cfg)
            ann = annotate(small, spots, skin, show_skin=args.show_skin)
            spot_dicts = [s.to_dict() for s in spots]

            ok, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with shared.cond:
                shared.jpeg, shared.spots = jpg.tobytes(), spot_dicts
                shared.cond.notify_all()

            streak = streak + 1 if spots else 0
            if body:
                body.found(len(spots) if streak >= 2 else 0)

            now = time.time()
            auto = (not args.no_autosave and streak >= args.stable_frames
                    and now - last_save > args.cooldown)
            if auto or shared.snap_request:
                # annotate full-res frame for the archive
                full_spots, full_skin = detect(frame, DetectorConfig(
                    px_per_mm=args.px_per_mm, contrast_thresh=args.contrast,
                    blackhat_kernel=int(cfg.blackhat_kernel / scale) | 1,
                    min_area_px=int(cfg.min_area_px / scale ** 2),
                    skin_erode_px=int(cfg.skin_erode_px / scale)))
                save_snapshot(Path(args.scans), frame, annotate(frame, full_spots, full_skin),
                              [s.to_dict() for s in full_spots])
                shared.snap_request, last_save = False, now
    except KeyboardInterrupt:
        pass
    finally:
        if body:
            body.close()
        src.close()
        server.shutdown()


if __name__ == "__main__":
    main()
