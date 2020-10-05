'''

This script collects various utility function that are used in the different
tasks for our model.

'''

import numpy as np

def kTrajectory (T, K = 3, Ar = (0.5, 2.0), Wr = (1, 2, 3, 5), offT = 0, norm = False):
    P = [];

    for k in range (K):
        A = np.random.uniform (*Ar, size = 4);
        W = np.array (Wr) * 2. * np.pi;
        F = np.random.uniform (0., 2. * np.pi, size = len (W));
        t = np.linspace (0., 1., num = T);

        p = 0.
        for a, w, f in zip (A, W, F):
            p += a * np.cos (t * w + f);

        P.append (p);
    P = np.array (P);

    # Here we normalize our trajectories
    P = P / np.max (P, axis = 1).reshape (K, 1) if norm else P;

    # Here we zero-out the initial offT entries of target
    P [:, :offT] = 0.;

    return P;

def kClock (T, K = 5):
    C = np.zeros ((K, T));

    for k, tick in enumerate (C):
        range = T // K;

        tick [k * range : (k + 1) * range] = 1;

    return C;

def sfilter (seq, itau = 0.5):
    filt_seq = np.zeros (seq.shape);

    for t, s in enumerate (seq.T):
        filt_seq [:, t] = filt_seq [:, t - 1] * itau + s * (1. - itau) if t > 0 else seq [:, 0];

    return filt_seq;

def dJ_rout (J_rout, targ, S_rout):
    Y = J_rout @ S_rout;

    return (targ - Y) @ S_rout.T;

def read (S, J_rout, itau_ro = 0.5):
    out = sfilter (S, itau = itau_ro);

    return J_rout @ out;
