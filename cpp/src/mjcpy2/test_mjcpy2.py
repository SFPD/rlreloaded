import mjcpy2
from control4.config import CTRL_ROOT
import numpy as np
import os.path as osp
np.set_printoptions(precision=3)
world = mjcpy2.MJCWorld2(osp.join(CTRL_ROOT,"domain_data/mujoco_worlds/humanoid.xml"))

d = world.GetModel()
x = np.concatenate([d["qpos0"].flatten(), np.zeros(d["nv"])])
u = np.ones(d["nu"])
import time
for i in xrange(20):
    world.Plot(x)
    combefore = world.GetCOMMulti(x.reshape(1,-1))
    x, f = world.Step(x, u)
    comafter = world.GetCOMMulti(x.reshape(1,-1))
    print comafter - combefore
    time.sleep(.01)
    data = world.GetData()
    if len(data["contacts"]) > 0:
        print "got contacts!"
        for c in data["contacts"]:
            frame = c['frame'].reshape(3,3)
            normal = frame[0]
            print "geom %i contacts geom %i"%(c['geom1'], c["geom2"]), normal

print "setfeats"
world.SetFeats(["cdof","cfrc_ext","contactdists"])
print "gonna stepping"
x, f = world.Step(x, u)
print "done stepping"
names_sizes = world.GetFeatDesc()
assert sum(size for (_,size) in names_sizes) == f.size

x[2] -= .2
# world.Idle(x)
eps=1e-8
while True:
    contacts = world.GetContacts(x)
    maxpen = - contacts["dist"][contacts["geom1"]==0].min()
    print "maxpen",maxpen
    if maxpen > eps:
        x[2] += maxpen
    else:
        break
    # world.Idle(x)

N = 1
X = np.random.randn(N, x.size)
U = np.random.randn(N, u.size)

X[:,2] += 1

F0 = np.empty((N, f.size))
Y0 = np.empty((N, x.size))

for i in xrange(N):
    Y0[i],F0[i] = world.Step(X[i],U[i])

Y1,F1 = world.StepMulti(X,U)

assert np.allclose(Y0,Y1)
assert np.allclose(F0,F1)
# world.Idle(x)
