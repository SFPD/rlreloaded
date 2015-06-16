#!/usr/bin/env python
from control4.algs.save_load_utils import load_agent_and_mdp
from control4.core.rollout import rollout
from tabulate import tabulate
import numpy as np
import pygame
from control3.pygameviewer import PygameViewer, pygame
from collections import namedtuple
from copy import deepcopy

path = []

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--load_idx",type=int,default=-1)
    parser.add_argument("--max_steps",type=int)
    parser.add_argument("--one_traj",action="store_true")
    args = parser.parse_args()
    agent, mdp, _hdf = load_agent_and_mdp(args.hdf,args.load_idx)


    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt    
    fig1,(ax0,ax1)=plt.subplots(2,1)
    fig2,(ax3)=plt.subplots(1,1)
    h = mdp.halfsize
    while True:
        path = []
        init_arrs, traj_arrs = rollout(mdp,agent,999999,save_arrs=["m","o","a"])
        m = np.concatenate([init_arrs["m"]]+traj_arrs["m"],axis=0)
        o = np.concatenate([init_arrs["o"]]+traj_arrs["o"],axis=0)
        a_na = np.concatenate(traj_arrs["a"])
        print "o:"
        print o
        print "m:"
        print m
        ax0.cla()
        ax0.plot(m)
        ax1.cla()
        ax1.plot(o)

        ax3.cla()
        x,y=np.array(init_arrs['x'].path).T
        ax3.plot(x,y,'bx-')
        ax3.axis([-h,h,-h,h])
        for (x,a) in zip(init_arrs['x'].path,a_na):
            ax3.add_artist(Ellipse(xy=x+a[0:2], width=2*a[2], height=2*a[3],alpha=0.2))

        plt.draw()
        plt.pause(0.01)
        plt.ginput()



