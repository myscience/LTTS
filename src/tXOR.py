import numpy as np
import utils as ut
from ltts import CuLTTS

# Here we define the parameters of our model
# ---------- SETTING FOR Np=2 -----------------
N = 400;
T = 120;
dt = 1 / T;
tau_m = 2. * dt;
tau_s = 2. * dt;
tau_ro = 10. * dt;
beta_s  = np.exp (-dt / tau_s);
beta_ro = np.exp (-dt / tau_ro);
sigma_teach = 6.;
sigma_input = 3.;
offT = 10;
dv = 1 / 5.;
alpha = .05;
alpha_rout = .05;
Vo = -4;
h = -1;
s_inh = 20;

# Here we build the dictionary of the simulation parameters
par = {'tau_m' : tau_m, 'tau_s' : tau_s, 'tau_ro' : tau_ro, 'beta_ro' : beta_ro,
   'dv' : dv, 'alpha' : alpha, 'alpha_rout' : alpha_rout, 'Vo' : Vo, 'h' : h, 's_inh' : s_inh,
   'N' : N, 'T' : T, 'dt' : dt, 'offT' : offT};

ltts = CuLTTS ((N, T), par, rand_init = False);

# Here we build the targets of the task
Np = 2;
Io = .9;
Inp, Out = ut.parityCode (N = Np);
Inp = Inp.reshape (2**Np, 1, -1) + Io;
Out = Out.reshape (2**Np, 1, -1);

Jteach = np.random.normal (0., sigma_teach, size = (N, 1));
Jinput = np.random.normal (0., sigma_input, size = (N, 1));

Iteach = [Jteach @ out for out in Out];
Iinput = [Jinput @ inp for inp in Inp];

Targ = [ltts.compute (It + Ii, init = np.zeros (N)) for It, Ii in zip (Iteach, Iinput)];
J_rout, track = ltts.train (Targ[:3], Iinput[:3], outs = Out[:3], mode = 'offline', epochs = 1000);

# Here we save this model
np.save ("tXOR.npy", np.array ([Inp, Out, Jteach, Jinput, J_rout, Targ,
                                ltts.J, par, track], dtype = np.object));
