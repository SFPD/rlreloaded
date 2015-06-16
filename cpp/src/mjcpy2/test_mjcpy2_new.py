import mjcpy2
from control4.config import CTRL_ROOT
from control4.mdps.mjc2 import MJCMDP
import numpy as np
import os.path as osp
np.set_printoptions(precision=3)
# world = mjcpy2.MJCWorld2(osp.join(CTRL_ROOT,"domain_data/mujoco_worlds/hopper-ball.xml"))
# world = mjcpy2.MJCWorld2(osp.join(CTRL_ROOT,"domain_data/mujoco_worlds/humanoid.xml"))

mdp = MJCMDP("3d_humanoid")
world = mdp.world

d = world.GetModel()
x = np.concatenate([d["qpos0"].flatten(), np.zeros(d["nv"])])
u = np.ones(d["nu"])

features = ["cinert","cvel","qfrc_actuation","cfrc_ext","contactdists"]
world.SetFeats(features)

print d["qpos0"]
print x

from control4.mdps.mjc2_info import *

model = world.GetModel()
actuators = get_actuators(model)
names = get_joint_names(model)

# x, f = world.Step(x, u)

# import time

# for i in range(500):
#     world.Plot(x)
#     x, f = world.Step(x, u)
#     x[1] += 0.01
#     features = extract_feature(world, f, "cvel")
    # print features[13]
#     print features
#     x, f = world.Step(x, u)
#     time.sleep(.01)


# import IPython
# IPython.embed()

# import sys
# sys.exit()

# print "nv is ", model['nv']
# print "nq is ", model['nq']

# print "length x is ", len(x)

actuator_to_dof = {}
k = 0
for act in model['dof_jntid'].squeeze():
    actuator_to_dof[act] = k
    k += 1

# import time
# for q in range(0, 27):
#     print "q ", q
#     oldval = x[q]
#     for i in range(500):
#         world.Plot(x)
        # x, f = world.Step(x, u)
#         x[q] += 0.01
        # x[jnt_idx] += 0.01
#        time.sleep(.01)
#     x[q] = oldval

# import sys
# sys.exit()

import math
import time
jnt_range = model["jnt_range"]
for a in range(model['nu']):
    jnt = model['actuator_trnid'][a,0]
    q = model['jnt_qposadr'][jnt,0]
    print "testing", names[a+1]
    print "a ", a
    print "q ", q
    left_lim = jnt_range[jnt,0]
    right_lim = jnt_range[jnt,1]
    print "joint limits ", left_lim, " ", right_lim
    rng = np.linspace(left_lim, right_lim, 500)
    oldval = x[q]
    for c in rng:
        # x, f = world.Step(x, u)
        x[q] = c
        world.Plot(x)
        time.sleep(.01)
    x[q] = oldval

import sys
sys.exit(0)

import math
import time

ctrlrange = model["actuator_ctrlrange"]
for a in range(model['nu']):
    print "testing", names[a+1]
    left_lim = ctrlrange[a,0]
    right_lim = ctrlrange[a,1]
    u = np.zeros(model['nu'])
    x_old = x.copy()
    u[a] = 100.0
    for c in range(500):
        x, f = world.Step(x, u)
        world.Plot(x)
        time.sleep(.01)
    print "opposite direction"
    u[a] = -100.0
    for c in range(500):
        x, f = world.Step(x, u)
        world.Plot(x)
        time.sleep(.01)
    x = x_old

import sys
sys.exit(0)

import math
import time
ctrlrange = model["actuator_ctrlrange"]
for a in range(model['nu']):
    jnt = model['actuator_trnid'][a,0]
    q = model['jnt_qposadr'][jnt,0]
    print "testing", names[a+1]
    left_lim = ctrlrange[a,0]
    right_lim = ctrlrange[a,1]
    print "actuator limits ", left_lim, " ", right_lim
    rng = np.linspace(left_lim/180.0*math.pi, right_lim/180.0*math.pi, 500)
    oldval = x[q]
    for c in rng:
        # x, f = world.Step(x, u)
        x[q] = c
        world.Plot(x)
        time.sleep(.01)
    x[q] = oldval

import sys
sys.exit(0)

import math
import time
ctrlrange = model["actuator_ctrlrange"]
for (joints, q) in actuator_to_dof.iteritems():
    if a == 0:
        continue
    print "a ", a
    print "testing ", names[a]
    print "actuator limits ", ctrlrange[a-1,0], " ", ctrlrange[a-1,1]
    oldval = x[q]
    rng = np.linspace(ctrlrange[a-1,0]/180.0*math.pi, ctrlrange[a-1,1]/180.0*math.pi, 500)
    for c in rng:
        # x, f = world.Step(x, u)
        # x[a+1] += 0.01 # for humanoid
        x[q] = c
        world.Plot(x)
        # x[jnt_idx] += 0.01
        time.sleep(.01)
    x[q] = oldval

import sys
sys.exit()

import time
for a in range(6, 27):
    print "testing ", names[model['dof_jntid'][a,0]]
    oldval = x[a+1]
    for i in range(500):
        world.Plot(x)
        # x, f = world.Step(x, u)
        x[a+1] += 0.01
        # x[jnt_idx] += 0.01
        time.sleep(.01)
    x[a+1] = oldval

def extract_feature(world, f, feature_name):
    startidx = 0
    for (name,size) in world.GetFeatDesc():
        if name == feature_name:
            return f[startidx:startidx+size]
            # print "using %s for impact cost"%name
            break
        else:
            startidx += size

import os
os.exit()