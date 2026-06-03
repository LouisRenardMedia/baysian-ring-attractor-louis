import numpy as np
from scipy.optimize import root_scalar
from scipy.special import i0, i1
from angle_utils import calc_angle

class Ring_Attractor:

    def __init__(self, N, dt, tau, kappa_phi, k_v, k_z, w_const, w_quad, kappa_0, phi_0, stoch_corr):

        self.N = N
        self.dt = dt
        self.tau = tau
        self.kappa_phi = kappa_phi
        self.k_v = k_v
        self.k_z = k_z
        self.w_const = w_const
        self.w_quad = w_quad
        self.kappa_0 = kappa_0
        self.phi_0 = phi_0
        self.stoch_corr = stoch_corr

        self.I_ext = self.xi_fun_inv(k_z * dt)

        self.mu = [phi_0]
        self.kappa = [kappa_0]


        self.r = []

        w_asym = k_v / (kappa_phi + k_v)
        w_sym = 1 / tau



        # vector of preferred HD
        self.phi = np.linspace(-np.pi, np.pi - (2 * np.pi) / N, N)

        # Set up weight matrix
        diff = self.phi[:, None] - self.phi[None, :]  # shape (N, N)
        self.W_sym = w_sym * (2 / N) * np.cos(diff)
        self.W_asym = (2 / N) * np.sin(diff) * w_asym
        self.W_const = 1 / N * np.ones((N, N)) * w_const

        # init activities
        self.r.append(kappa_0 * np.cos(self.phi - phi_0))



    def RNN_step(self, dy=0, z=0):
        """" Runs a recurrent neural network dynamics, with parameters matched to
        approximate the circKF.

        Input:
        dt          - time step
        w_sym      - even recurrent connectivity
        w_asym       - odd recurrent connectivity
        tau         - decay time constant
        w_quad      - quadratic weight
        stoch_corr  - stochastic correction (additional decay due to Ito conversion)
        dy          - increment observation

        Output:
        mu      - mean estimate after update
        kappa   - certainty estimate after update """

        f_act = lambda x: np.maximum(0, x)

        # set up all-to-all summation
        M = np.pi / self.N * np.ones([self.N, self.N])

        # add Wiener process if there is neural noise
        # if sigma_N != 0:
        #     dW = np.sqrt(dt) * np.random.randn(int(T / dt), N)
        # else:
        #     dW = np.zeros((int(T / dt), N))

        # run network filter
        W = self.W_sym + self.W_asym * (dy / self.dt) + self.W_const


        self.r.append((self.r[-1]
                  - self.stoch_corr * self.r[-1] * self.dt  # stochastic correction
                  - 1 / self.tau * self.r[-1] * self.dt  # decay
                  + np.dot(W, self.r[-1]) * self.dt  # angular velocity integration, recurrent stabilization
                  - self.w_quad * np.dot(M, f_act(self.r[-1])) * self.r[-1] * self.dt  # quadratic inhibition
                  + self.I_ext * np.cos(self.phi - z)))  # absolute heading info (external input)
        # + sigma_N * dW[i]))


        # decode stochastic variables
        basis = np.array([np.cos(self.phi), np.sin(self.phi)])  # (2, N)

        theta = (2 / self.N) * (basis @ self.r[-1])  # (2,)
        # theta[0] = κ·cos(μ) and theta[1] = κ·sin(μ)

        mu = np.arctan2(theta[1], theta[0])
        kappa = np.linalg.norm(theta)



        self.mu.append(mu)
        self.kappa.append(kappa)

    def run_RNN(self, prev_angle=None, frames_since_detection=1, c=None):
        if c is not None:
            angle = calc_angle(c[0])

            if prev_angle != np.inf:
                dy = (((prev_angle - angle) + np.pi) % (
                            2 * np.pi) - np.pi) / frames_since_detection  # Wrapped ngular displacement in last frame
                self.RNN_step(dy=dy, z=angle)

            else:
                self.RNN_step()
        else:

            self.RNN_step()
            angle = None

        return angle

    def A_Bessel(kappa):
        """Computes the ratio of Bessel functions."""
        r = i1(kappa) / i0(kappa)
        return r

    def xi_fun_inv(self, dt):
        """Computes the inverse of the ratio of Bessel functions by root-finding."""
        f = lambda alpha: alpha * self.A_Bessel(alpha) - dt
        sol = root_scalar(f, bracket=[0.001, 50], method='brentq')
        alpha = sol.root
        return alpha





