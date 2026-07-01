import threading
import time
import numpy as np
import board
import busio
import adafruit_bno055

import Distance


class IMUReader:
    def __init__(self, bearing):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)
        self._sum = 0
        self._count = 0
        self.heading = 0
        self.previous_heading = 0
        self.distanceTracker = Distance.DistanceTracker(0.4)
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.bearing = bearing


    def _run(self):
        while not self._stop.is_set():
            gyro = self.sensor.gyro
            a = self.sensor.linear_acceleration
            a_east = a[0]
            a_north = a[1]
            if all(x is not None for x in (gyro, a_east, a_north)):
                with self.lock:
                    self.heading = self.sensor.euler[0]
                    self.distanceTracker.predict(a_north, a_east, self.bearing, 0.01)
                    w = self.distanceTracker.effective_omega(-gyro[2])
                    self._sum += w
                    self._count += 1

            time.sleep(0.01)

    def get(self, bearing):
        with self.lock:
            avg = self._sum / self._count if self._count else 0.0
            if not self.heading: self.heading = 0.0
            dtheta = ((np.radians(self.previous_heading - self.heading) + np.pi) % (2* np.pi)) - np.pi
            self.previous_heading = self.heading
            self._sum = 0.0
            self._count = 0
            self.bearing = bearing

        return avg, dtheta

    def getdistance(self):
        return self.distanceTracker.x[0]

    def stop(self):
        self._stop.set()