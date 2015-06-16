from control3.common import *

theano.config.optimizer="None"

for mdpname in ["cartpole_barto","cartpole_doya","pendulum","mjc:3swimmer"]:
    print "checking",mdpname
    mdp = get_mdp(mdpname)
    mdp.make_funcs()
    X = np.random.randn(10,mdp.state_dim()).astype(floatX)
    U = np.random.randn(10,mdp.ctrl_dim()).astype(floatX)

    assert np.allclose(mdp.fs_dyn(X,U), np.array([mdp.f_dyn(x,u) for (x,u) in zip(X,U)]))
    assert np.allclose(mdp.fs_cost(X,U), np.array([mdp.f_cost(x,u) for (x,u) in zip(X,U)]))
    assert np.allclose(mdp.fs_done(X), np.array([mdp.f_done(x) for x in X]))
