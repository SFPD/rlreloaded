from control4.misc.console_utils import call_and_print,Message
import os.path as osp
from control4.core.rollout import rollout
from control4.algs.save_load_utils import get_mdp,construct_agent,load_agent_and_mdp
from control4.maths.discount import discount
from control4.config import floatX
import numpy as np
from collections import namedtuple
from sklearn import linear_model
from pylab import *
import time
from pandas.stats.moments import ewma

# mdp = get_mdp("atari:breakout",kws={"obs_mode":"ram"})
# agent = construct_agent({"agent_module":"control4.agents.random_atari_agent","policy_cfg":{}},mdp)


fname = "/Users/xyz/Data/rlreloaded_data/results/ram-atari-frameskip+vf1/frameskip2+lam0.99-breakout_RUN00.h5"
agent, mdp, hdf = load_agent_and_mdp(fname,2)


gamma = 1.0
num_trajs = 4
max_steps=99999

Path = namedtuple("Path",['o','c','v'])

save_arrs = ["o","c",'v']
with Message("Doing rollouts"):
    paths = []
    for i_path in xrange(num_trajs):
        if i_path % 20 == 0:
            print "%i/%i done"%(i_path,num_trajs)
        init,traj = rollout(mdp, agent, max_steps,save_arrs=save_arrs)
        o = np.concatenate([init["o"]]+traj["o"][:-1])
        c = np.concatenate(traj["c"])
        v = np.concatenate(traj['v'])
        paths.append(Path(o,c,v))


# Regression
oall = np.concatenate([path.o for path in paths],axis=0).astype(floatX)
vall = np.concatenate([discount(path.c.sum(axis=1), gamma) for path in paths])

gamma = 1.0
# vf_lam = hdf['params/lam'].value
vf_lam = 0.01
# lam = hdf['params/lam'].value

oall = np.concatenate([path.o for path in paths],axis=0).astype(floatX)
goodinds = np.flatnonzero(oall.std(axis=0)>0)
# goodinds = [57]
volds = [path.v*0 for path in paths]
oall = oall[:,goodinds]
K = oall.T.dot(oall)
K[np.arange(K.shape[0]), np.arange(K.shape[0])] += 1e-4

# for i in xrange(1):
#     vtargs = []
#     for (path,vold) in zip(paths,volds):
#         c = path.c.sum(axis=1)[:-1]
#         delta = c + gamma*vold[1:] - vold[:-1]
#         vtarg = vold[:-1] + discount(delta,gamma*vf_lam)
#         vtarg = np.concatenate([vtarg,[0]])
#         vtargs.append(vtarg)
#     volds = vtargs
#     vtargall = np.concatenate(vtargs)
#     w = np.linalg.solve(K, oall.T.dot(vtargall))
#     print "using vf_lam < 1"
#     f = lambda o: o.dot(w)
#     # model = linear_model.RidgeCV()
#     # model.fit(oall,vtargall)
#     from control4.maths.numeric import explained_variance_1d
#     vpred = f(oall)
#     print "iteration",i,"ev:",explained_variance_1d(vpred,vall),"predvar",vpred.var(),"actualvar",vall.var()


w = np.linalg.lstsq(oall,vall)[0]
print "using vf_lam = 1"
f = lambda o: o.dot(w)
lam = hdf['params/lam'].value



# ylim(-0.4, 5)
plt.clf()
path = paths[2]
c = path.c[:-1].sum(axis=1)
v = f(path.o[:,goodinds])
# v = path.v
# print "plotting actual vf"
lives = path.o[:-1,57]
plot(-c)
plot(-v[:-1]/10)
delta = (c + gamma*v[1:] - v[:-1])



plot(-delta)
plot(lives*15)
plot(discount(-delta, lam))
legend([r"$c_t$",r"$v_t$",r"$\delta$","lives","adv"])
show()

plt.figure(2)
plt.clf()
vs = []
cs = []
deltas = []
def plot_callback(arrs):
    mdp.plot(arrs)
    # vs.append(arrs['v'].sum())
    vs.append(f(arrs['o'][:,goodinds]).sum())
    cs.append(arrs['c'].sum())
    if len(vs) > 1:
        delta = cs[-1] + gamma * vs[-2] - vs[-1]
        deltas.append(delta)
        if len(vs) % 4 == 0:
            plt.cla()
            plt.plot(cs,'b')
            plt.plot(np.array(vs)/10.0,'g')
            plt.plot(deltas,'r')
            plt.ylim(-10,5)
            plt.pause(.0001)
    
_,arrs = rollout(mdp,agent,max_steps,save_arrs=("c",),callback=plot_callback)    
