from control4.algs.save_load_utils import load_agent_and_mdp
import h5py
import os.path as osp,os
import numpy as np
import argparse
from control4.mdps.atari import load_library
from control4.misc.console_utils import yes_or_no
load_library()
from control4.mdps.atari import emuGetRam,emuGetImage
from control4.misc.h5_utils import setitem_maybe_compressed

def rollout2(mdp, agent, max_length, save_arrs=()):
    """
    Do single rollout from random initial states and save results to a list
    Returns
    --------
    init_arrs: mapping name -> row vector, specifying initial condition
    traj_arrs: mapping name -> list of row vectors, specifying state at t=0,1,...,T-1    
    """
    init_arrs = {}
    init_arrs.update(mdp.initialize_mdp_arrays())
    init_arrs.update(agent.initialize_lag_arrays())

    traj_arrs = {name:[] for name in save_arrs}
    traj_arrs["ram"] = []
    traj_arrs["image"] = []

    cur_arrs = init_arrs.copy()
    for _ in xrange(max_length):
        emuGetImage(mdp.e,mdp.raw_img_buf)
        traj_arrs["image"].append(mdp.raw_img_arr.copy())
        emuGetRam(mdp.e,mdp.ram_buf)
        traj_arrs["ram"].append(mdp.ram_arr.copy())
        cur_arrs.update(agent.call(cur_arrs))
        cur_arrs.update(mdp.call(cur_arrs))
        for name in save_arrs: traj_arrs[name].append(cur_arrs[name])
        if cur_arrs.get("done",False):
            break
    return (init_arrs, traj_arrs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")
    parser.add_argument("--load_idx",type=int,default=-1)
    parser.add_argument("--max_steps",type=int,default=999999)
    parser.add_argument("--num_traj",type=int,default=100)

    args = parser.parse_args()
    agent, mdp, _hdf = load_agent_and_mdp(args.infile,args.load_idx)


    if osp.exists(args.outfile):
        if yes_or_no("%s exists. delete?"%args.outfile):
            os.unlink(args.outfile)
        else:
            raise IOError

    outfile = h5py.File(args.outfile,"w")

    for i_traj in xrange(args.num_traj):
        if i_traj % 10 == 0:
            print i_traj,args.num_traj
        grpname = "%.4d"%i_traj
        grp = outfile.create_group(grpname)
        init,arrs = rollout2(mdp,agent,args.max_steps,save_arrs=("o","c","a","b","u"))
        for key in arrs:
            if key == "o":
                setitem_maybe_compressed(grp, 'o', np.concatenate([init['o']] + arrs['o']))
            elif key in ("image","ram"):
                setitem_maybe_compressed(grp,key,np.array(arrs[key]))
            else:
                setitem_maybe_compressed(grp,key,np.concatenate(arrs[key]))

if __name__ == "__main__":
    main()