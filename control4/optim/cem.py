import numpy as np


def cem(f,th_mean,batch_size,n_iter,elite_frac, initial_std=1.0, extra_std=0.0, std_decay_time=1.0):
    r"""
    Noisy cross-entropy method
    http://dx.doi.org/10.1162/neco.2006.18.12.2936
    http://ie.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf
    Incorporating schedule described on page 4 (also see equation below.)

    Inputs
    ------

    f : function of one argument--the parameter vector
    th_mean : initial distribution is theta ~ Normal(th_mean, initial_std)
    batch_size : how many samples of theta per iteration
    n_iter : how many iterations
    elite_frac : how many samples to select at the end of the iteration, and use for fitting new distribution
    initial_std : standard deviation of initial distribution
    extra_std : "noise" component added to increase standard deviation.
    std_decay_time : how many timesteps it takes for noise to decay

    \sigma_{t+1}^2 =  \sigma_{t,elite}^2 + extra_std * Z_t^2
    where Zt = max(1 - t / std_decay_time, 10 , 0) * extra_std.
    """
    n_elite = int(np.round(batch_size*elite_frac))

    th_std = np.ones(th_mean.size)*initial_std

    for iteration in xrange(n_iter):

        extra_var_multiplier = max((1.0-iteration/std_decay_time),0) # Multiply "extra variance" by this factor
        sample_std = np.sqrt(th_std + np.square(extra_std) * extra_var_multiplier)

        ths = np.array([th_mean + dth for dth in  sample_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        assert ys.ndim==1
        elite_inds = ys.argsort()[:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.var(axis=0)
        yield {"ys":ys,"th":th_mean,"ymean":ys.mean()}
