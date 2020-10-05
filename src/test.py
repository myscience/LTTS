import numpy as np
import utils as ut
from kim import KIM

# Here we define the parameters of our model
N = 500;
T = 1000;
dt = 1 / T;
tau_m = 8. * dt;
tau_s = 2. * dt;
tau_ro = 20. * dt;
beta_s  = np.exp (-dt / tau_s);
beta_ro = np.exp (-dt / tau_ro);
sigma_teach = 2.;
sigma_clock = 4.;
offT = 20;
dv = 1 / 20.;
alpha = 0.01;
alpha_rout = 0.02;
Vo = -4;
h = -4;
s_inh = 20;

# Here we build the dictionary of the simulation parameters
par = {'tau_m' : tau_m, 'tau_s' : tau_s, 'tau_ro' : tau_ro, 'beta_ro' : beta_ro,
   'dv' : dv, 'alpha' : alpha, 'alpha_rout' : alpha_rout, 'Vo' : Vo, 'h' : h, 's_inh' : s_inh,
   'N' : N, 'T' : T, 'dt' : dt, 'offT' : offT};

kim = KIM ((N, T), par);

# Here we set up the 3-Trajectory Task
P = ut.kTrajectory (T, K = 3, offT = offT, norm = True);
C = ut.kClock (T, K = 5);

Jteach = np.random.normal (0., sigma_teach, size = (N, 3));
Jclock = np.random.normal (0., sigma_clock, size = (N, 5));

Iteach = np.matmul(Jteach, P);
Iclock = np.matmul(Jclock, C);

S_init = np.zeros (N);
S_targ = kim.compute (Iteach + Iclock, init = S_init);

# Here we train our model
J_rout, track = kim.train (S_targ, Iclock, out = P, epochs = 1000, track = True);

# Here we save the results of our training
np.save ("Trained Model.npy", np.array ([P, C, Iteach, Iclock, J_rout, S_targ,
                                        kim.J, par, track], dtype = np.object));
