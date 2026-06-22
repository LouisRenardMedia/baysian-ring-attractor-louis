import threading
import time


class Recorder:
    def __init__(self, cap):
        self.last_frame_id = None
        self.frame_id = 0
        self.cap = cap
        self.frame = None
        self.timestamp = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.frames_since_detection = 0

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.time()  # timestamp at capture, not at processing
                    self.frames_since_detection += 1
                    self.frame_id += 1

    def get(self):
        with self.lock:
            if self.frame_id == self.last_frame_id:
                return None, None, self.frames_since_detection

            count = self.frames_since_detection
            self.frames_since_detection = 0
            self.last_frame_id = self.frame_id
            return self.frame, self.timestamp, count

    def stop(self):
        self._stop.set()

    def frame_count_set(self, count):
        with self.lock:
            self.frames_since_detection += count -1