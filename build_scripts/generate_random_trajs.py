#!/usr/bin/env python
from control3 import mjc2_mdps
from control3.common import *
import os.path as osp
N_TOTAL_MIN = 5000
MAX_TRAJ_LEN = 1000

def do_random_rollout(mdp):
    lo,hi = mdp.ctrl_bounds()
    iss = mdp.initial_state_sampler()

    xcuos = []

    x = iss.get_1()
    t=0
    while True:

        u = np.random.uniform(lo,hi).astype(floatX)
        y,c,o = mdp.f_dyn_cost_obs(x,u)

        xcuos.append((x,c,u,o))
        x = y
        if mdp.f_done(x):
            print "trajectory reached terminal state after %i timesteps"%len(xcuos)
            break
        if len(xcuos) >= MAX_TRAJ_LEN: 
            print "stopped rollout after %i timesteps"%len(xcuos)
            break
        t+=1
    return xcuos

def gen_mdp_data(basename):
    print "MDP: %s"%basename
    mdp = mjc2_mdps.MJCMDP(basename)
    mdp.make_funcs()
    fname = osp.expandvars(osp.join("$CTRL_DATA/misc/mdp_random_trajs","mjc2:"+basename+".h5"))
    mkdirp(osp.dirname(fname))
    hdf = h5py.File(fname,"w")    

    n_sofar = 0
    while n_sofar < N_TOTAL_MIN:
        grp = hdf.create_group("%.3i"%len(hdf))
        xcuos = do_random_rollout(mdp)
        n_sofar += len(xcuos)
        xs,cs,us,obs = zip(*xcuos)
        grp["xs"] = xs
        grp["cs"] = cs
        grp["us"] = us
        grp["os"] = obs


def main():
    np.set_printoptions(precision=3)

    basenames = mjc2_mdps.mdp_names()
    for basename in basenames:
        gen_mdp_data(basename)


if __name__ == "__main__":
    main()
