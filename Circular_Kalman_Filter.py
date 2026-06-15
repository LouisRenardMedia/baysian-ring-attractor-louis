import numpy as np

from scipy.special import i0, i1
from angle_utils import calc_angle


class CKF:
    def __init__(self, kappa_phi, dt, k_z, k_v):
        '''
        Initialize circular Kalman filter object, initializes mu and kappa array representing estimate history
        with angle 0 and certainty 1

        Input:
        kappa_phi   - inverse diffusion constant of hidden state process
        dt          - time step
        k_z         - reliability of single HD observation
        k_v         - precision of increment observation
        '''

        self.mu = [0.]
        self.kappa = [1.]
        self.kappa_phi = kappa_phi
        self.dt = dt
        self.k_z = k_z
        self.k_v = k_v





    def kalman_step(self, z=0, dy=0, k_z=0, k_v=0):
        '''
        compute single Euler–Maruyama step of Circular Kalman filter, appends direction and certainty estimates
        to mu array and kappa array fields

        Input:
        z           - HD observation
        k_z         - reliability of single HD observation (set to 0 for no observation)
        dy          - increment observation
        k_v         - precision of increment observation (set to 0 for no observation)
        '''

        # update (on natural parameters -> robust in discrete time)
        if k_z != 0:
            az, bz = self.polar_to_euclidean(k_z, z)
            a, b = self.polar_to_euclidean(self.kappa[-1], self.mu[-1])
            a = a + az
            b = b + bz
            mu, kappa = self.euclidean_to_polar(a, b)

        # prediction (include increment observations)
        if k_v != 0:
            dmu_pred = k_v / (self.kappa_phi + k_v) * dy
        else:
            dmu_pred = 0
        dkappa_pred = - 1 / 2 * 1 / (self.kappa_phi + k_v) * self.kappa[-1] * self.f_kappa(kappa) * self.dt

        mu_out = self.mu[-1] + dmu_pred
        mu_out = ((mu_out + np.pi) % (2 * np.pi)) - np.pi  # mu in[-pi,pi]
        kappa_out = self.kappa[-1] + dkappa_pred

        self.mu.append(mu_out)
        self.kappa.append(kappa_out)

    def run_exp_step(self, prev_angle=None, frames_since_detection=1, c=None):
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

                self.kalman_step(angle, dy, k_z=self.k_z, k_v=self.k_v)

            else:
                self.kalman_step(k_v=0, k_z=0)

        else:
            # Setting k_v and k_z to 0 ignores any incoming information in the absence of a circle
            self.kalman_step(k_v=0, k_z=0)
            angle = None
        return angle

    def polar_to_euclidean(self, r, phi):
        """ Converts a polar coordinate with radius r and angle phi to Cartesian coordinates. """
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y

    def euclidean_to_polar(self, x, y):
        """ Converts a Cartesian to polar coordinates. """
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return phi, r

    def A_Bessel(self, kappa):
        """Computes the ratio of Bessel functions."""
        r = i1(kappa) / i0(kappa)
        return r

    def f_kappa(self, kappa):
        """ Computes the precision decay function in the circKF. """
        f = self.A_Bessel(kappa) / (kappa - self.A_Bessel(kappa) - kappa * self.A_Bessel(kappa) ** 2)
        return f


