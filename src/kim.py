import numpy as np
import utils as ut
from tqdm import trange
from optimizer import Adam

class KIM:
    """
        This is the Kinetic Ising Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, shape, par):
        # This are the network size N and the temporal sequence lenght T
        self.N, self.T = shape;

        self.dt = 1. / self.T;
        self.itau_m = self.dt / par['tau_m'];
        self.itau_s = np.exp (-self.dt / par['tau_s']);

        self.dv = par['dv'];

        # This is the network connectivity matrix
        self.J = np.zeros ((self.N, self.N));

        # Remove self-connections
        np.fill_diagonal (self.J, 0.);

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh);

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (shape) * h;

        # Membrane potential
        self.H = np.ones (shape) * par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (shape);
        self.S_hat = np.zeros (shape);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def compute (self, inp, init, Tmax = None, Vo = None, dv = None):
        '''
            This function is used to compute the output of our model given an
            input.

            Args:
                inp : numpy.array of shape (N, T), where N is the number of neurons
                      in the network and T is the time length of the input.

                init: numpy.array of shape (N, ), where N is the number of
                      neurons in the network. It defines the initial condition on
                      the spikes. Should be in range [0, 1]. Continous values are
                      casted to boolean.

            Keywords:
                Tmax: (default: None) Optional time legth of the produced output.
                      If not provided default is self.T

                Vo  : (default: None) Optional initial condition for the neurons
                      membrane potential. If not provided defaults to external
                      field h for all neurons.

                dv  : (default: None) Optional different value for the dv param
                      to compute the sigmoid activation.

        '''
        # Check correct input shape
        assert inp.shape[0] == self.N;

        dv = self.dv if dv is None else dv;

        itau_m = self.itau_m;
        itau_s = self.itau_s;

        if isinstance (init, str) and init == 'random':
            self.S [:, 0] = np.where (np.random.uniform (0., 1., size = self.N) > 1. - m, 1., 0.);

        elif isinstance (init, np.ndarray):
            # Here we check that provided initial condition is valid for spikes
            # initialization: should be in range [0, 1].
            assert np.min (init) >= 0 and np.max (init) <= 1.;

            self.S [:, 0] = init.astype (np.bool).copy ();

        else:
            raise ValueError ('Unknown init condition {} in compute.'.format (type (init)));

        if Vo is not None:
            self.H [:, 0] = Vo.copy ();

        self.S_hat [:, 0] = self.S [:, 0] * itau_s;
        _T = self.T - 1 if Tmax is None else Tmax - 1;

        for t in range (_T):
            self.S_hat [:, t] = self.S_hat [:, t - 1] * itau_s + self.S [:, t] * (1. - itau_s) if t > 0 else self.S_hat [:, 0];

            self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:, t] + inp [:, t] + self.h [:, t])\
                                                              + self.Jreset @ self.S [:, t];

            self.S [:, t + 1] = self._sigm (self.H [:, t + 1], dv = dv) - 0.5 > 0.;

        return self.S [:, :_T + 1].copy ();

    def train (self, targ, inp, mode = 'online', epochs = 500, par = None,
                out = None, Jrout = None, track = False):
        '''
            This is the main function of the model: is used to trained the system
            given a target and and input. Two training mode can be selected:
            (offline, online). The first uses the exact likelihood gradient (which
            is non local in time, thus non biologically-plausible), the second is
            the online approx. of the gradient as descrived in the article.

            Args:
                targ: numpy.array of shape (N, T), where N is the number of neurons
                      in the network and T is the time length of the sequence.

                inp : numpy.array of shape (N, T) that collects the input signal
                      to neurons.

            Keywords:
                mode : (default: online) The training mode to use, either 'offline'
                       or 'online'.

                epochs: (default: 500) The number of epochs of training.

                par   : (default: None) Optional different dictionary collecting
                        training parameters: {dv, alpha, alpha_rout, beta_ro, offT}.
                        If not provided defaults to the parameter dictionary of
                        the model.

                out   : (default: None) Output target trajectories, numpy.array
                        of shape (K, T), where K is the dimension of the output
                        trajectories. This parameter should be specified if either
                        Jrout != None or track is True.

                Jrout : (default: None) Pre-trained readout connection matrix.
                        If not provided, a novel matrix is built and trained
                        simultaneously with the recurrent connections training.
                        If Jrout is provided, the out parameter should be specified
                        as it is needed to compute output error.

                track : (default: None) Flag to signal whether to track the evolution
                        of output MSE over training epochs. If track is True then
                        the out parameters should be specified as it is needed to
                        compute output error.

        '''
        assert (targ.shape == inp.shape);

        par = self.par if par is None else par;

        dv = par['dv'];

        itau_m = self.itau_m;
        itau_s = self.itau_s;

        sigm = self._sigm;

        alpha = par['alpha'];
        alpha_rout = par['alpha_rout'];
        beta_ro = par['beta_ro'];
        offT = par['offT'];

        self.S [:, 0] = targ [:, 0].copy ();
        self.S_hat [:, 0] = self.S [:, 0] * itau_s;

        Tmax = np.shape (targ) [-1];
        dH = np.zeros ((self.N, self.T));

        track = np.zeros (epochs) if track else None;

        opt_rec = Adam (alpha = alpha, drop = 0.9, drop_time = 100 * Tmax if mode == 'online' else 100);

        if Jrout is None:
            S_rout = ut.sfilter (targ, itau = beta_ro);
            J_rout = np.random.normal (0., 0.1, size = (out.shape[0], self.N));
            opt = Adam (alpha = alpha_rout, drop = 0.9, drop_time = 20 * Tmax if mode == 'online' else 20);

        else:
            J_rout = Jrout;

        for epoch in trange (epochs, leave = False, desc = 'Training {}'.format (mode)):
            if Jrout is None:
                # Here we train the readout
                dJrout = (out - J_rout @ S_rout) @ S_rout.T;
                J_rout = opt.step (J_rout, dJrout);

            # Here we train the network
            for t in range (Tmax - 1):
                self.S_hat [:, t] = self.S_hat [:, t - 1] * itau_s + targ [:, t] * (1. - itau_s) if t > 0 else self.S_hat [:, 0];

                self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:, t] + inp [:, t] + self.h [:, t])\
                                                                  + self.Jreset @ targ [:, t];

                dH [:, t + 1] = dH [:, t]  * (1. - itau_m) + itau_m * self.S_hat [:, t];

                if mode == 'online':
                    dJ = np.outer (targ [:, t + 1] - sigm (self.H [:, t + 1], dv = dv), dH [:, t + 1]);
                    self.J = opt_rec.step (self.J, dJ);

                    np.fill_diagonal (self.J, 0.);

            if mode == 'offline':
                dJ = (targ - sigm (self.H, dv = dv)) @ dH.T;
                self.J = opt_rec.step (self.J, dJ);

                np.fill_diagonal (self.J, 0.);

            # Here we track MSE
            if track is not None:
                S_gen = self.compute (inp, init = np.zeros (self.N));
                track [epoch] = np.mean ((out - J_rout @ ut.sfilter (S_gen,
                                                itau = beta_ro))[:, offT:]**2.);

        return (J_rout, track) if Jrout is None else track;
