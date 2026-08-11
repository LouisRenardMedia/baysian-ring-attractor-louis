# robot_toy.py
# Run on the Pi via SSH. Control with WASD + Q/E to turn, X to stop, ESC to quit.
# Press R to toggle raw video recording (no unwarp).

import sys
import tty
import termios
import threading
import time
import datetime
import queue
import cv2
from pathlib import Path
import RPi.GPIO as GPIO
import Adafruit_PCA9685

# ── Motor setup ─────────────────────────────────────────────────────────────
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1, IN2 = 23, 24
IN3, IN4 = 27, 22
ENA, ENB = 0, 1

for pin in (IN1, IN2, IN3, IN4):
    GPIO.setup(pin, GPIO.OUT)

# ── Speed constants ──────────────────────────────────────────────────────────
DRIVE_SPEED = 1300
TURN_SPEED  = 1400
SPIN_SPEED  = 1400

# ── Low-level motor primitives ───────────────────────────────────────────────
def _set_motors(left, right):
    GPIO.output(IN1, GPIO.HIGH if left  < 0 else GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW  if left  < 0 else GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH if right < 0 else GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW  if right < 0 else GPIO.HIGH)
    pwm.set_pwm(ENA, 0, abs(int(left)))
    pwm.set_pwm(ENB, 0, abs(int(right)))

def stop():
    for pin in (IN1, IN2, IN3, IN4):
        GPIO.output(pin, GPIO.LOW)
    pwm.set_pwm(ENA, 0, 0)
    pwm.set_pwm(ENB, 0, 0)

# ── Commands ─────────────────────────────────────────────────────────────────
COMMANDS = {
    'w': ('Forward',    lambda: _set_motors( DRIVE_SPEED,  DRIVE_SPEED)),
    's': ('Backward',   lambda: _set_motors(-DRIVE_SPEED, -DRIVE_SPEED)),
    'a': ('Arc left',   lambda: _set_motors( TURN_SPEED // 2,  TURN_SPEED)),
    'd': ('Arc right',  lambda: _set_motors( TURN_SPEED,  TURN_SPEED // 2)),
    'q': ('Spin left',  lambda: _set_motors(-SPIN_SPEED,  SPIN_SPEED)),
    'e': ('Spin right', lambda: _set_motors( SPIN_SPEED, -SPIN_SPEED)),
    'x': ('Stop',       stop),
}
QUIT_KEY = '\x1b'  # ESC

_go_home_queue = queue.Queue(maxsize=1)
_go_home_stop = threading.Event()
_go_home_thread = None


def go_home(mu, kappa, t=0.2):
    move_time = 0.01 + 0.02 * kappa
    turn_time = 0.01 + 0.01 * kappa
    if mu > t:
        COMMANDS.get('e')[1]()
        time.sleep(turn_time)
        COMMANDS.get('w')[1]()
        time.sleep(move_time)
    elif mu < -t:
        COMMANDS.get('q')[1]()
        time.sleep(turn_time)
        COMMANDS.get('w')[1]()
        time.sleep(move_time)
    else:
        COMMANDS.get('w')[1]()
        time.sleep(move_time)
    time.sleep(0.2)


def submit_go_home_target(mu, kappa):
    try:
        mu = float(mu)
        kappa = float(kappa)
    except (TypeError, ValueError):
        return

    try:
        _go_home_queue.put_nowait((mu, kappa))
    except queue.Full:
        try:
            _go_home_queue.get_nowait()
        except queue.Empty:
            pass
        _go_home_queue.put_nowait((mu, kappa))


def _go_home_loop():
    try:
        while not _go_home_stop.is_set():
            try:
                mu, kappa = _go_home_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            go_home(mu, kappa)
            _go_home_queue.task_done()
    finally:
        stop()


def start_go_home_control():
    global _go_home_thread
    if _go_home_thread is not None and _go_home_thread.is_alive():
        return _go_home_thread

    _go_home_stop.clear()
    _go_home_thread = threading.Thread(target=_go_home_loop, daemon=True)
    _go_home_thread.start()
    return _go_home_thread


def stop_go_home_control():
    _go_home_stop.set()
    stop()


def experimentPath():
    COMMANDS.get('w')[1]()
    time.sleep(2)
    COMMANDS.get('d')[1]()
    time.sleep(4)
    COMMANDS.get('a')[1]()
    time.sleep(8)
    COMMANDS.get('q')[1]()
    time.sleep(2)
    COMMANDS.get('w')[1]()
    time.sleep(3)
    COMMANDS.get('e')[1]()
    time.sleep(4.5)
    COMMANDS.get('x')[1]()
    time.sleep(4)
    COMMANDS.get('a')[1]()
    time.sleep(5.5)
    COMMANDS.get('w')[1]()
    time.sleep(6)
    COMMANDS.get('x')[1]()

# ── Camera / recorder ────────────────────────────────────────────────────────
CAM_INDEX     = 0
CAM_FPS       = 30
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

class Recorder:
    def __init__(self):
        self._cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(CAM_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open camera")

        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._writer = None
        self._recording = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                if self._recording and self._writer is not None:
                    self._writer.write(frame)

    def start_recording(self):
        with self._lock:
            if self._recording:
                return None
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(RECORDINGS_DIR / f"recording_{ts}.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._writer = cv2.VideoWriter(filename, fourcc, CAM_FPS, (self._w, self._h))
            self._recording = True
        return filename

    def stop_recording(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            if self._writer is not None:
                self._writer.release()
                self._writer = None

    def release(self):
        self._stop_event.set()
        self.stop_recording()
        self._cap.release()

# ── Raw key capture ──────────────────────────────────────────────────────────
def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    # print("Robot toy controller")
    # print("  W/S   — forward / backward")
    # print("  A/D   — arc left / right")
    # print("  Q/E   — spin left / right in place")
    # print("  X     — stop")
    # print("  R     — toggle recording")
    # print("  ESC   — quit")
    # print()

    # recorder = Recorder()

    try:
        while True:
            key = get_key().lower()

            if key == QUIT_KEY:
                print("\nQuitting.")
                break

            # if key == 'r':
            #     if not recorder._recording:
            #         fname = recorder.start_recording()
            #         print(f"  Recording started → {fname}")
            #     else:
            #         recorder.stop_recording()
            #         print("  Recording stopped.")
            #     continue

            if key in COMMANDS:
                label, action = COMMANDS[key]
                # print(f"  {label}")
                action()

    finally:
        stop()
        # recorder.release()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
