import numpy as np

from circularFiltering import vM_Projection
from detect_red_circles import calc_angle


class CKF:
    def __init__(self, kappa_phi, dt, k_z, k_v):
        self.mu = [0.]
        self.kappa = [1.]
        self.kappa_phi = kappa_phi
        self.dt = dt
        self.k_z = k_z
        self.k_v = k_v





    def kalman_step(self, mu, kappa, z=0, dy=0, k_z=0, k_v=0):
        '''
        compute single Euler–Maruyama step of Circular Kalman filter
        '''
        return vM_Projection(mu, kappa, self.kappa_phi, z=z, dy=dy, dt=self.dt, kappa_z=k_z, kappa_v=k_v)

    def run_CircKF(self, prev_angle=None, frames_since_detection=1, c=None):
        '''
        Run one step of the circKF depending on the parameters inputed.

        prev_angle: angle of previously detected circle
        dt:         time step
        c:          circle array containing x co-ordinate necessery for angle caluclation

        return: angle, angle of detected circle, only used when calling with a circle detected
        '''

        # When a circle is detected, pass information to circKF call
        if c is not None:
            angle = calc_angle(c[0])

            if prev_angle != np.inf:
                dy = (((prev_angle - angle) + np.pi) % (2 * np.pi) - np.pi) / frames_since_detection  # Wrapped angular displacement in last frame

                mean, kappa = self.kalman_step(self.mu[-1], self.kappa[-1], angle, dy, k_z=self.k_z, k_v=self.k_v)

            else:
                mean, kappa = self.kalman_step(self.mu[-1], self.kappa[-1], k_v=0, k_z=0)

        else:
            # Setting k_v and k_z to 0 ignores any incoming information in the absence of a circle
            mean, kappa = self.kalman_step(self.mu[-1], self.kappa[-1], k_v=0, k_z=0)
            angle = None

        self.mu.append(mean)
        self.kappa.append(kappa)
        return angle