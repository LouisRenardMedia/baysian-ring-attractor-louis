import cv2
import json
import threading
import time
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_HERE = Path(__file__).parent

DEFAULT_CFG = {
    "x1": 79, "y1": 42, "x2": 515, "y2": 478,
    "inner_radius": 70,
    "outer_radius": 218,
    "output_width": 1300,
    "output_height": 200,
    "p": 1.0,
    "angle_offset": -1.5707963267948966
}


class UnwarpCamera:
    """
    Opens the Pi camera and unwarps each frame on read().

    Usage in another file:
        from stream_unwarp import UnwarpCamera
        cam = UnwarpCamera()
        ok, pano = cam.read()   # drop-in like cv2.VideoCapture
        cam.release()

        # or as a context manager:
        with UnwarpCamera() as cam:
            for pano in cam.frames():
                ...

        # optional MJPEG stream to laptop:
        cam.start_stream(port=8080)   # http://<pi-ip>:8080/stream
    """

    def __init__(self, cam_index=0, cfg_path=None, calib_path=None):
        cfg_path = Path(cfg_path) if cfg_path else _HERE / "unwarp_cfg.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                self._cfg = json.load(f)
        else:
            self._cfg = DEFAULT_CFG
            with open(cfg_path, "w") as f:
                json.dump(self._cfg, f, indent=2)
            print(f"[UnwarpCamera] wrote default config to {cfg_path} — edit to match your mirror")

        self._map1 = self._map2 = None
        if calib_path and Path(calib_path).exists():
            data = np.load(calib_path)
            self._calib_mtx = data["mtx"]
            self._calib_dist = data["dist"]
            print(f"[UnwarpCamera] loaded calibration from {calib_path}")
        elif calib_path:
            print(f"[UnwarpCamera] calib file not found: {calib_path} — skipping undistortion")

        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        fps=cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        self._cap = cap
        self._jpeg = b""
        self._jpeg_lock = threading.Lock()

    # ------------------------------------------------------------------
    # image processing
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_power(img, cx, cy, r_in, r_out, W, H, p, angle_offset):
        r_norm = np.linspace(1.0, 0.0, H, dtype=np.float32)
        r_corr = r_in + (r_norm ** p) * (r_out - r_in)

        theta = np.linspace(0.0, 2 * np.pi, W, endpoint=False, dtype=np.float32) - angle_offset
        cos_t = np.cos(theta)[None, :]
        sin_t = np.sin(theta)[None, :]
        R = r_corr[:, None]

        map_x = (cx + R * cos_t).astype(np.float32)
        map_y = (cy + R * sin_t).astype(np.float32)

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _undistort(self, img):
        if not hasattr(self, "_calib_mtx"):
            return img
        h, w = img.shape[:2]
        if self._map1 is None:
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(self._calib_mtx, self._calib_dist, (w, h), alpha=0)
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self._calib_mtx, self._calib_dist, None, new_mtx, (w, h), cv2.CV_16SC2
            )
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)

    def _crop_and_unwrap(self, img):
        img = self._undistort(img)
        cfg = self._cfg
        cropped = img[cfg["y1"]:cfg["y2"], cfg["x1"]:cfg["x2"]]
        cx = cropped.shape[1] // 2
        cy = cropped.shape[0] // 2

        pano = self._unwrap_power(
            cropped, cx, cy,
            cfg["inner_radius"],
            cfg["outer_radius"],
            cfg["output_width"],
            cfg["output_height"],
            cfg.get("p", 1.14),
            cfg.get("angle_offset", -np.pi / 2),
        )
        pano = cv2.flip(pano, 1)

        return pano

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def read(self):
        """Return (True, unwrapped_bgr) or (False, None)."""
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        return True, self._crop_and_unwrap(frame)

    def frames(self):
        """Generator that yields unwrapped BGR frames until the camera fails."""
        while True:
            ok, pano = self.read()
            if not ok:
                time.sleep(0.005)
                continue
            yield pano
    
    def read_crop(self):
        """Return (True, cropped_bgr) or (False, None) — no unwarp applied."""
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        cfg = self._cfg
        frame = self._undistort(frame)
        return True, frame[cfg["y1"]:cfg["y2"], cfg["x1"]:cfg["x2"]]
    
    def start_crop_stream(self, port=8081):
        """Start a background MJPEG server for the cropped (pre-unwarp) image."""
        jpeg_ref = [b""]
        lock = threading.Lock()

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with lock:
                            jpg = jpeg_ref[0]
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _push_loop():
            while True:
                ok, crop = self.read_crop()
                if not ok:
                    time.sleep(0.005)
                    continue
                enc_ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if enc_ok:
                    with lock:
                        jpeg_ref[0] = jpg.tobytes()

        threading.Thread(target=_push_loop, daemon=True).start()
        server = HTTPServer(("0.0.0.0", port), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[UnwarpCamera] crop stream at http://<pi-ip>:{port}/stream")

    def start_stream(self, port=8080):
        """Start a background MJPEG server. Laptop opens http://<pi-ip>:{port}/stream."""
        jpeg_ref = [self._jpeg]
        lock = self._jpeg_lock

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with lock:
                            jpg = jpeg_ref[0]
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _push_loop():
            while True:
                ok, pano = self.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                enc_ok, jpg = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if enc_ok:
                    with self._jpeg_lock:
                        jpeg_ref[0] = jpg.tobytes()

        threading.Thread(target=_push_loop, daemon=True).start()
        server = HTTPServer(("0.0.0.0", port), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[UnwarpCamera] streaming at http://<pi-ip>:{port}/stream")

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
