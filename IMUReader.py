import threading
import time

import board
import busio
import adafruit_bno055

class IMUReader:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)
        self.gyro_z = 0.0  # angular velocity around vertical axis (rad/s)
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            gyro = self.sensor.gyro  # returns (x, y, z) in rad/s
            if gyro is not None:
                with self.lock:
                    self.gyro_z = gyro[2]  # z axis = yaw rate
            time.sleep(0.01)  # 100Hz, faster than camera

    def get(self):
        with self.lock:
            return self.gyro_z

    def stop(self):
        self._stop.set()