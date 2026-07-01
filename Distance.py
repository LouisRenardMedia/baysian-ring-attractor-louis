import sys

import numpy as np


class DistanceTracker:
    """
    Estimates distance to target using:
    - Linear acceleration (IMU) as process input
    - Circle radius from vision as noisy distance observation

    State: [d, v] where d = distance, v = speed along bearing
    """

    def __init__(self, d0, process_noise=0.1, obs_noise=0.2):
        """
        d0              - initial distance estimate
        focal_length    - camera focal length in pixels
        real_radius     - real radius of ball in metres
        process_noise   - Q: trust in dynamics model
        obs_noise       - R: trust in circle size observation
        """

        # state [d, v]
        self.x = np.array([d0, 0.0])

        self.perp_v = 0

        # covariance
        self.P = np.eye(2) * 1.0

        # process noise
        self.Q = np.array([[0.0, 0.0],
                           [0.0, process_noise]])

        # observation noise
        self.R = obs_noise

    def predict(self, a_north, a_east, phi, dt):
        """
        a       - linear acceleration from IMU (m/s^2)
        phi     - robot bearing to taget (rad)
        dt      - time step
        """

        # component of acceleration along bearing to target
        a = np.sqrt(a_north ** 2 + a_east ** 2)
        a_along = a * np.cos(phi)
        a_across = a * np.sin(phi)
        self.perp_v = a_across * dt
        # state transition
        # d_new = d - v*dt  (moving toward target reduces distance)
        # v_new = v + a_along*dt
        F = np.array([[1.0, -dt],
                      [0.0, 1.0]])

        B = np.array([0.0, dt])

        self.x = F @ self.x + B * a_along

        if abs(self.x[1]) < 0.06:
            self.x[1] *= 0.6
        if abs(self.x[1]) < 0.001:
            self.x[1] = 0

        self.P = F @ self.P @ F.T + self.Q
        #print("distance: ",self.x[0],"velocity: ",self.x[1],"acceleration: ", a_along)




    def update(self, target_pixel):
        """
        pixel_radius - radius of detected ball in pixels
        """


        d_obs = self.distanceObs(target_pixel)
        if d_obs == None:
            return

        H = np.array([1.0, 0.0])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S

        innovation = d_obs - H @ self.x
        self.x = self.x + K * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P

    def effective_omega(self, omega_imu):
        """
        Compute corrected angular velocity accounting for translational motion.
        omega_imu   - raw yaw rate from IMU (rad/s)
        theta       - bearing to target from ring attractor (rad)
        phi         - robot heading (rad)
        """
        d, v = self.x
        if d < 0.01:  # avoid division by zero
            return omega_imu
        correction =  self.perp_v / d
        # print("originalIMU: ", omega_imu, "// correctedIMU: ", omega_imu + correction)
        return omega_imu #+ correction

    def distanceObs(self, target_pixel):
        pixels = np.array([82, 97, 147, 199])
        meters = np.array([0.60, 0.30,0.15, 0.0])
        # Clamp outside the range
        if target_pixel <= pixels[0]:
            return None
        if target_pixel >= pixels[-1]:
            return None

        return np.interp(target_pixel, pixels, meters)

    def radius_to_unwrapped_y(self, r, r_in, r_out, H, p):
        return (H - 1) * (
                1 - ((r - r_in) / (r_out - r_in)) ** (1 / p)
        )
