import numpy as np
from scipy.optimize import root_scalar
from scipy.special import ive

class BayesianRingAttractor:

    def __init__(self, N, dt, tau, kappa_phi, k_v, k_z, w_const, w_quad, kappa_0, phi_0, stoch_corr):
        '''
        initialize activity with mean phi_0 and certainty kappa_0

        Fields:
        N           - number of neurons
        dt          - time step
        tau         - decay time constant
        k_z         - HD observation certainty
        k_v         - angular velocity observation certainty
        w_const     - uniform excitatory connection strength applied between all neurons
        w_quad      - quadratic weight
        phi_0       - initial mean estimate
        kappa_0     - initial certainty estimate
        stoch_corr  - stochastic correction (additional decay due to Ito conversion)
        I_ext       - external input (computer through xi_fun_inv())
        mu[]        - array containing mean direction history
        kappa[]     - array containing certainty history
        r[]         - array containing activity vector history
        W_sym       - symetrical recurrent connectivity exciting neurons close together
        W_asym      - Asymmetric recurrent connectivity moving the bump around with angular velocity
        W_const     - uniform connectivity matrix
        '''
        self.N = N
        self.dt = dt
        self.tau = tau
        self.kappa_phi = kappa_phi
        if np.isscalar(k_v):
            self.k_v = np.array([k_v], dtype=float)
        else:
            self.k_v = np.array(k_v, dtype=float)
        self.k_z = k_z
        self.w_const = w_const
        self.w_quad = w_quad
        self.kappa_0 = kappa_0
        self.phi_0 = phi_0
        self.stoch_corr = stoch_corr

        if k_z == 0:
            self.I_ext = 0
        else:
            self.I_ext = self.xi_fun_inv(k_z * dt)

        self.mu = [phi_0]
        self.kappa = [kappa_0]


        self.r = []

        # vector of preferred HD
        self.phi = np.linspace(-np.pi, np.pi, N, endpoint=False)

        # Set up weight matrix
        self.diff = self.phi[:, None] - self.phi[None, :]  # shape (N, N)



        self.W_asym = []
        k_v_total = np.sum(self.k_v)
        w_asym = [k_v / (kappa_phi + k_v_total) for k_v in self.k_v]
        w_sym = 1 / tau + 1 / (kappa_phi + k_v_total)

        for i in range(len(self.k_v)):
            self.W_asym.append((2 / N) * np.sin(self.diff) * w_asym[i])

        self.W_sym = w_sym * (2 / N) * np.cos(self.diff)
        self.W_const = 1 / N * np.ones((N, N)) * w_const

        # init activities
        self.r.append(kappa_0 * np.cos(self.phi - phi_0))


    def step(self, dy=0, z=None, k_z=None):
        """" Runs recurrent neural network dynamics, with parameters matched to
        approximate the circKF.

        Input:
        dy          - angular velocity
        z           - HD observations
        """

        f_act = lambda x: np.maximum(0, x)

        # set up all-to-all summation
        M = np.pi / self.N * np.ones([self.N, self.N])
        z_cancel = 1
        if z is None:
            z = 0
            z_cancel = 0
        if dy is None:
            dy = 0

        if k_z is not None:
            if k_z == 0:
                self.I_ext = 0
            else:
                self.I_ext = self.xi_fun_inv(k_z * self.dt)

        # add Wiener process if there is neural noise
        # if sigma_N != 0:
        #     dW = np.sqrt(dt) * np.random.randn(int(T / dt), N) not array needed
        # else:
        #     dW = 0

        # run filter
        if np.isscalar(dy):
            dy = [dy]

        W = self.W_sym + self.W_const
        # print("kv: ", self.k_v)

        k_v_total = sum(self.k_v)
        w_asym = [k_vi / (self.kappa_phi + k_v_total) for k_vi in self.k_v]

        self.W_asym = []
        for i in range(len(self.k_v)):
            self.W_asym.append((2 / self.N) * np.sin(self.diff) * w_asym[i])

        for i in range(len(dy)):
            W += self.W_asym[i] * dy[i]

        self.r.append((self.r[-1]
                  - self.stoch_corr * self.r[-1] * self.dt  # stochastic correction
                  - 1 / self.tau * self.r[-1] * self.dt  # decay
                  + np.dot(W, self.r[-1]) * self.dt  # angular velocity integration, recurrent stabilization
                  - self.w_quad * np.dot(M, f_act(self.r[-1])) * self.r[-1] * self.dt  # quadratic inhibition
                  + z_cancel * self.I_ext * np.cos(self.phi - z)))  # absolute heading info (external input)
        # + sigma_N * dW))


        # decode stochastic variables
        basis = np.array([np.cos(self.phi), np.sin(self.phi)])  # (2, N)

        theta = (2 / self.N) * (basis @ self.r[-1])  # (2,)
        # theta[0] = κ·cos(μ) and theta[1] = κ·sin(μ)

        mu = np.arctan2(theta[1], theta[0])
        kappa = np.linalg.norm(theta)


        self.mu.append(mu)
        self.kappa.append(kappa)



    def A_Bessel(self, kappa):
        """Computes the ratio of Bessel functions."""
        r = ive(1, kappa) / ive(0, kappa)
        return r

    def xi_fun_inv(self, dt):
        """Computes the inverse of the ratio of Bessel functions by root-finding."""
        if not np.isfinite(dt) or dt <= 0:
            return 0

        f = lambda alpha: alpha * self.A_Bessel(alpha) - dt
        lower = 0.0
        upper = max(50.0, dt + 1.0)
        while f(upper) < 0:
            upper *= 2

        sol = root_scalar(f, bracket=[lower, upper], method='brentq')
        alpha = sol.root
        return alpha

    def update_weights(self, omega, eta=0.07, sigma=0.05, k_vt=None, floor_frac=0.05):
        """
        Multiplicative Hebbian trust update
        for 3 angular-velocity sensors.

        k_v     - array [k_v1, k_v2, k_v3]  current weights
        omega   - array [w1, w2, w3]        current sensor readings
        eta     - learning rate (log-domain)
        sigma   - agreement bandwidth (how close = "close")
        k_vt    - target total weight (defaults to current sum of k)
        floor_frac - minimum weight, as a fraction of k_vt, before
                     the update is applied (keeps recovery possible)
        """
        k_v = self.k_v

        if k_vt is None:
            k_vt = sum(k_v)

        # pairwise agreement in [0,1], 1 = identical readings
        a12 = np.exp(-((omega[0] - omega[1]) ** 2) / (2 * sigma ** 2))
        a13 = np.exp(-((omega[0] - omega[2]) ** 2) / (2 * sigma ** 2))
        a23 = np.exp(-((omega[1] - omega[2]) ** 2) / (2 * sigma ** 2))

        # net agreement signal per sensor (positive = reinforced, negative = penalized)
        s = np.array([
            a12 / 2 + a13 / 2 - a23,  # sensor 1: rewarded by agreeing w/ 2,3; hurt if 2,3 agree without it
            a12 / 2 - a13 + a23 / 2,  # sensor 2
            -a12 + a13 / 2 + a23 / 2  # sensor 3
        ])

        k_v = np.maximum(k_v, floor_frac * k_vt)  # floor before exponentiating
        k_v = k_v * np.exp(eta * s)  # multiplicative update
        k_v *= k_vt / k_v.sum()  # renormalize to conserve total
        self.k_v = k_v
        print(k_v)




