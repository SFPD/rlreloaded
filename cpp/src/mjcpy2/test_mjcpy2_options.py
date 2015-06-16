import mjcpy2
from control4.config import CTRL_ROOT
import numpy as np
import os.path as osp
import time

world = mjcpy2.MJCWorld2(osp.join(CTRL_ROOT,"domain_data/mujoco_worlds/humanoid.xml"))

d = world.GetModel()
x = np.concatenate([d["qpos0"].flatten(), np.zeros(d["nv"])])
u = 100 * np.ones(d["nu"])

features = ["cinert","cvel","qfrc_actuation","cfrc_ext","contactdists"]
world.SetFeats(features)

print d["qpos0"]
print x

from control4.mdps.mjc2_info import *

model = world.GetModel()
actuators = get_actuators(model)
names = get_joint_names(model)

data = world.GetData()

option = world.GetOption()

option['expdist'] = 1.0

world.SetOption(option)

print world.GetOption()