#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--hdf")
parser.add_argument("--agent_module")
parser.add_argument("--mdp_name")
parser.add_argument('--mdp_kws')
parser.add_argument("--max_steps",type=int,default=100)
parser.add_argument("--load_idx",type=int,default=-1)
parser.add_argument("--snapshot_index","-si",type=int,default=-1)
args = parser.parse_args()
from pylab import * #pylint: disable=W0622
from control4.core.rollout import rollout
from control4.algs.save_load_utils import construct_agent,get_mdp,load_agent_and_mdp
import time
assert bool(args.hdf is None) != bool(args.agent_module is None)
if args.hdf:    
    agent, mdp, _hdf = load_agent_and_mdp(args.hdf,args.load_idx)
elif args.agent_module:
    mdp = get_mdp(args.mdp_name)
    agent = construct_agent({"agent_module":args.agent_module},mdp)

# ipython --pylab -i /Users/xyz/Synced/Proj/control/scripts/diagnostics/view_mujoco_timeseries.py  -- --agent_module=control4.agents.random_continuous_agent --mdp_name=mjc2:walker2d

while True:
    def plot_callback(arrs):
        mdp.plot(arrs)
        time.sleep(.02)
    init_arrs,traj_arrs = rollout(mdp, agent, args.max_steps, save_arrs=("o","x","u","c"),callback=plot_callback)
    names_sizes = mdp.obs_names_sizes()
    nplot = len(names_sizes)
    plt.figure(1)
    plt.clf()
    obs = np.concatenate(traj_arrs["o"])

    start=0
    for (i,(name,size)) in enumerate(names_sizes):

        assert obs.shape[1] == sum(s for (n,s) in mdp.obs_names_sizes())

        print name,obs[:,start:start+size].std(axis=0)
        # Raw
        plt.subplot(nplot,1,i+1)
        plt.gca().set_title(name)        
        plt.plot(obs[:,start:start+size])
        ylo,yhi = plt.ylim()
        plt.ylim( ylo + .1*(ylo - yhi), yhi + .1*(yhi-ylo) )         
        start += size

    cost = np.concatenate(traj_arrs["c"])
    plt.figure()
    plt.plot(cost, lw=3.0)
    plt.legend(mdp.cost_names())

    assert start == obs.shape[1]
    plt.show()
    raw_input("press enter")


