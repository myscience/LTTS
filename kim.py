import numpy as np
from tqdm import trange

class KIM:
    """
        This is the Kinetic Ising Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A novel
        (actually very akin to full-FORCE) training algorithm is used to perform
        temporal sequence learning.
    """

    def __init__ (self, shape, h = 0., s_inh = 100.):
        # This are the network size N and the temporal sequence lenght T
        self.N, self.T = shape;

        self.dt = 1. / self.T;
        self.tau_H = self.dt * 8.;

        # This is the network connectivity matrix
        self.J = np.random.uniform (0., .1, size = (self.N, self.N));

        # Erase self-connections
        np.fill_diagonal (self.J, -s_inh);

        # This is the external field
        self.h = h;

        # This is the intrinsic potential
        self.H = np.random.uniform (-0.5, 0.5, size = shape);

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (shape);
        self.S_hat = np.zeros (shape);

    @classmethod
    def _sigm (cls, x):
        return np.where (x > 0., 1. / (1. + np.exp (-x)), np.exp (x) / (1. + np.exp (x)));

    def compute (self, inp, init = 0., beta = .3):
        tau = self.dt / self.tau_H;

        self.S [:, 0] = init;
        self.S_hat [:, 0] = init * beta;

        for t in range (self.T - 1):
            self.S_hat [:, t] = self.S_hat [:, t - 1] * beta + self.S [:, t - 1] * (1. - beta) if t > 0 else self.S_hat [:, 0];

            self.H [:, t + 1] = self.H [:, t] * (1. - tau) + tau * (np.matmul (self.J, self.S_hat [:, t]) + inp [:, t] + self.h [:, t]);

            self.S [:, t + 1] = self._sigm (self.H [:, t + 1]) - 0.5 > 0.;

        return self.S.copy ();

    def train (self, targ, inp, opt, epochs = 500, beta = .3):
        tau = self.dt / self.tau_H;

        self.S [:, 0] = targ [:, 0];

        for epoch in trange (epochs, leave = False):
            dH = 0;

            for t in range (self.T - 1):
                self.S_hat [:, t] = self.S_hat [:, t - 1] * beta + targ [:, t - 1] * (1. - beta) if t > 0 else self.S [:, 0];

                self.H [:, t + 1] = self.H [:, t] * (1. - tau) + tau * (np.matmul (self.J, self.S_hat [:, t]) + inp [:, t] + self.h [:, t]);

                self.S [:, t + 1] = self._sigm (self.H [:, t + 1]) - 0.5 > 0.;

                dH = dH * (1. - tau) + tau * self.S_hat [:, t];
                dJ = np.tensordot (targ [:, t + 1] - self._sigm (self.H [:, t + 1]), dH, axes = 0);

                self.J = opt.step (self.J, dJ);
