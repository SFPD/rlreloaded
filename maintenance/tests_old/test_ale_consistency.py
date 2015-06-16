import alepy
import numpy as np
import os.path as osp
from control3 import CTRL_ROOT
# import cv2

world = alepy.AtariWorld(osp.join(CTRL_ROOT,"domain_data/atari_roms/space_invaders.bin"))


for j in xrange(5):
    x0 = world.GetInitialState(np.random.randint(0,50))
    u0 = np.array([0],'uint8')
    y,r,o,d = world.Step(x0,u0)

    for i in xrange(3):
        y1,r1,o1,d1 = world.Step(x0,u0)
        assert (y==y1).all() and (r==r1) and (np.array(o)==np.array(o1)).all()


        nsteps = np.random.randint(10)
        x = x0
        for t in xrange(nsteps):
            u = np.array([np.random.randint(0,10)],dtype='uint8')
            x,_,_,_ = world.Step(x,u)
