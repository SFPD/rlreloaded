import mjcpy
import numpy as np
from control3 import CTRL_ROOT
import os.path as osp
from control3.common_util import colorize
world = mjcpy.MJCWorld(osp.join(CTRL_ROOT,"cpp/3rdparty/mjc/mjcdata/3swimmer.bin"))

X,U = np.random.rand(9,10), np.random.rand(9,2)
x,u = np.random.rand(10), np.random.rand(2)

world.SetActuatedDims([3,4])

for i in xrange(10): assert np.allclose(world.Step(x,u),world.Step(x,u))
for i in xrange(10): assert np.allclose(world.StepMulti(X,U),world.StepMulti(X,U))
assert np.allclose(world.StepMulti(X,U), np.array([world.Step(x1,u1) for (x1,u1) in zip(X,U)]))

try:
    import numdifftools as ndt

    _, anjacx, anjacu = world.StepJacobian(x,u)
    numjacx = ndt.Jacobian(lambda x1:world.Step(x1,u))(x)
    numjacu = ndt.Jacobian(lambda u1:world.Step(x,u1))(u)
    assert np.allclose(anjacx,numjacx)
    assert np.allclose(anjacu,numjacu)
except ImportError:
    print colorize("don't have numdifftools. skipping test","yellow")

# result = world.Step(x,u)
# world.Plot(x,u)
# from time import sleep
# for i in xrange(10):
#     x = np.random.randn(10)
#     world.Plot(x,u)
#     sleep(.2)


