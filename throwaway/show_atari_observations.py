#!/usr/bin/env python
from control4.algs.save_load_utils import load_agent_and_mdp,construct_agent,get_mdp
from control4.core.rollout import rollout
from control4.algs.alg_params import string2dict
from tabulate import tabulate
import time
import numpy as np
import cv2

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_module")
    parser.add_argument("--mdp_name")
    parser.add_argument("--hdf")
    parser.add_argument("--load_idx",type=int,default=-1)
    parser.add_argument("--max_steps",type=int,default=999999)
    parser.add_argument("--stochastic",action="store_true")
    parser.add_argument("--pause_every",action="store_true")
    parser.add_argument("--one_traj",action="store_true")
    parser.add_argument("--fps",type=float,default=20)
    parser.add_argument("--mdp_kws",default="obs_mode=image")
    args = parser.parse_args()

    assert bool(args.hdf) != bool(args.agent_module)
    if args.hdf:    
        agent, mdp, _hdf = load_agent_and_mdp(args.hdf,args.load_idx)
    elif args.agent_module:
        mdp = get_mdp(args.mdp_name,kws=string2dict(args.mdp_kws))
        agent = construct_agent({"agent_module":args.agent_module},mdp)

    delay = 1/args.fps if args.fps>0 else 0
    def plot_callback(arrs):
        img = arrs["o"].reshape(mdp.img_shape())[:3,:,:].transpose(1,2,0)
        img = cv2.resize(img, (img.shape[1]*5,img.shape[0]*5),interpolation=cv2.INTER_NEAREST)        
        cv2.imshow("blah",img)
        cv2.waitKey(10)

    while True:
        _,arrs = rollout(mdp,agent,args.max_steps,save_arrs=("c",),callback=plot_callback)    
        c_tv = np.concatenate(arrs['c'])
        csum_v = c_tv.sum(axis=0)
        print tabulate(zip(mdp.cost_names(),csum_v) + [("total",c_tv.sum())])
        print "done after",c_tv.shape[0],"steps"
        if args.one_traj: break
        raw_input("press enter to continue")

if __name__ == "__main__":
    main()