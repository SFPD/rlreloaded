from __future__ import division
import numpy as np
import cycontrol
N=2
K=3

p_nk = np.random.randn(N,K)

np.random.seed(10)

n_trials = 100

for i in xrange(100):
    z = np.random.rand()
    p_k = np.random.rand(K).astype('float32')
    p_k /= p_k.sum()
    assert cycontrol.categorical1(p_k, z) == cycontrol.categorical2(p_k[None,:], np.array([z],'float32'))

p_k = np.array([.2, .3, .5],'float32')

n_trials = 100000
draws = np.array([cycontrol.categorical1(p_k, z) for z in np.random.rand(n_trials)])
counts = np.bincount(draws)
ep_k=counts / counts.sum().astype('float')
var_k = (p_k * (1-p_k)) / np.sqrt(n_trials)
assert ((p_k - ep_k).__abs__() < np.sqrt(var_k)*3).all()