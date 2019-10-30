import numpy as np
import matplotlib.pyplot  as plt

from matplotlib import rc
from kim import KIM
from optimizer import Adam

rc('font',**{'family':'serif', 'size': 16});
rc('text', usetex=True)


def buildTarget (T, K = 3):
    P = [];

    for k in range (K):
        A = np.random.uniform (0.2, 1.5, size = 4);
        W = np.array ([1, 2, 3, 5]) * 2. * np.pi;
        F = np.random.uniform (0., 2. * np.pi, size = 4);
        t = np.linspace (0., T * dt, num = T);

        p = 0.
        for a, w, f in zip (A, W, F):
            p += a * np.cos (t * w + f);

        P.append (p);

    return np.array (P);

def buildClock (T, ticks = 5):
    C = np.zeros ((ticks, T));

    for k, tick in enumerate (C):
        range = T // ticks;

        tick [k * range : (k + 1) * range] = 1;

    return C;

def dJ_rout (J_rout, targ, S_rout):
    Y = np.matmul (J_rout, S_rout);

    return np.matmul (targ - Y, S_rout.T);

def read (S, J_rout, beta = 0.8):
    S_rout = np.zeros (S.shape);

    S_rout [:, 0] = S [:, 0];

    for t, s in enumerate (S.T):
        S_rout [:, t] = S_rout [:, t - 1] * beta_rout + s * (1. - beta_rout) if t > 0 else S_rout [:, 0];

    return np.matmul (J_rout, S_rout);


N = 500;
T = 1000;
shape = (N, T);

dt = 1. / T;

K, ticks = 3, 5;
P = buildTarget (T, K = K);
C = buildClock (T, ticks = ticks);

inp_P = np.matmul (np.random.normal (0., 10., size = (N, K)), P);
inp_C = np.matmul (np.random.normal (0., 2., size = (N, ticks)), C);

m = 0.1;
h = -0.5 * np.ones (shape);

kim = KIM (shape, h = h);

S_init = np.where (np.random.uniform (0., 1., size = N) > 1. - m, 1., 0.);
S_targ = kim.compute (inp_P + inp_C, init = S_init);

# Here we train the output layer
J_rout = np.random.normal (0., .01, size = (K, N));

S_rout = np.zeros (shape);
beta_rout = 0.8;
S_rout [:, 0] = S_targ [:, 0];

for t, s in enumerate (S_targ.T):
    S_rout [:, t] = S_rout [:, t - 1] * beta_rout + s * (1. - beta_rout) if t > 0 else S_rout [:, 0];

adam = Adam ();
J_rout = adam.optimize (dJ_rout, P, S_rout, init = J_rout, t_max = 5000);

Y = np.matmul (J_rout, S_rout);
MSE = np.mean ((P - Y)**2.);

fig, ax = plt.subplots (figsize = (15, 10));

recon, _, _ = ax.plot (Y.T, c = 'b');
targ, _, _ = ax.plot (P.T, 'r--');

ax.set_xlabel ('Time');
ax.set_ylabel ('Amplitude');
ax.set_title ('Trained Readout MSE = {:.4f}'.format (MSE));
ax.legend ((targ, recon), ['Target', 'Reconstruced']);
fig.savefig ('../results/Trained_Readout.png', dpi = 200);
plt.show ();

fig, axes = plt.subplots (nrows = 3, figsize = (15, 13));

C = [['orange', 'darkorange'], ['dodgerblue', 'steelblue'], ['forestgreen', 'darkgreen']]
for ax, y, p, c in zip (axes, Y, P, C):
    recon, = ax.plot (y, c = c [0]);
    targ, = ax.plot (p, '--', c = c [1]);
    ax.set_xlabel ('Time');
    ax.set_ylabel ('Amplitude');
    ax.legend ((targ, recon), ['Target', 'Reconstruced']);

axes [0].set_title ('Trained Readout MSE = {:.4f}'.format (MSE));
fig.savefig ('../results/Trained_Readout_splitted.png', dpi = 200);
plt.show ();

kim.train (S_targ, inp_C, Adam (), epochs = 300);

# Here we test the generated sequence
S_gen = kim.compute (inp_C, init = S_init);

Y = read (S_gen, J_rout);
MSE = np.mean ((P - Y)**2.);

fig, ax = plt.subplots (figsize = (15, 10));

recon, _, _ = ax.plot (Y.T, c = 'b');
targ, _, _ = ax.plot (P.T, 'r--');

ax.set_xlabel ('Time');
ax.set_ylabel ('Amplitude');
ax.set_title ('Generated Readout MSE = {:.4f}'.format (MSE));
ax.legend ((targ, recon), ['Target', 'Generated']);
fig.savefig ('../results/Generated_Readout.png', dpi = 200);
plt.show ();

fig, axes = plt.subplots (nrows = 3, figsize = (15, 13));

C = [['orange', 'darkorange'], ['dodgerblue', 'steelblue'], ['forestgreen', 'darkgreen']]
for ax, y, p, c in zip (axes, Y, P, C):
    recon, = ax.plot (y, c = c [0]);
    targ, = ax.plot (p, '--', c = c [1]);
    ax.set_xlabel ('Time');
    ax.set_ylabel ('Amplitude');
    ax.legend ((targ, recon), ['Target', 'Generated']);

axes [0].set_title ('Generated Readout MSE = {:.4f}'.format (MSE));
fig.savefig ('../results/Generated_Readout_splitted.png', dpi = 200);
plt.show ();


plt.show ();
