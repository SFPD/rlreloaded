from collections import defaultdict
from control4.agents.pixel_atari_agent import PixelAtariAgent
from control4.agents.ram_atari_agent import RamAtariAgent
from control4.algs.diagnostics import write_diagnostics
from control4.algs.save_load_utils import get_mdp,setup_outfile,fetch_file,is_save_iter,save_agent_snapshot
from control4.config import floatX,resolve_cfg_loc
from control4.config import setup_logging
from control4.maths import symbolic
from control4.misc.console_utils import Message,Timers
from control4.misc.var_collection import VarCollection
from control4.optim.adaptive_descent import adaptive_descent
from control4.optim.cg_optimize import cg_optimize
from control4.optim.lbfgs import lbfgs
from tabulate import tabulate
import argparse
import numpy as np
import theano.tensor as TT, theano #pylint: disable=F0401
import time,json,h5py,sys

parser = argparse.ArgumentParser()
parser.add_argument("--datafile")
parser.add_argument("--policy_cfg")
parser.add_argument("--metadata")
parser.add_argument("--n_traj",type=int)
parser.add_argument("--n_iter",type=int,default=100)
parser.add_argument("--seed",type=int)
parser.add_argument("--outfile")
parser.add_argument("--opt_method",choices=["lbfgs","cg","cg_with_rescaling","sgd","none"],default="lbfgs")
parser.add_argument("--obs_mode",choices=["image","ram"],default="ram")
parser.add_argument("--use_color",type=int,default=0)
parser.add_argument("--renormalize",type=int,default=0)
parser.add_argument("--snapshot_period",type=int,default=20)
parser.add_argument("--normalize",type=int,default=0)
args = parser.parse_args()

params = args.__dict__
np.set_printoptions(precision=3)
np.random.seed(args.seed)
setup_logging()

with open(resolve_cfg_loc(args.policy_cfg),"r") as fh: 
    policy_cfg = json.load(fh)
vf_cfg = None
mdp = get_mdp("atari:pong",kws={"obs_mode":args.obs_mode,"use_color":args.use_color})
agent = (PixelAtariAgent if args.obs_mode=="image" else RamAtariAgent)(mdp, policy_cfg, vf_cfg)
agent.pprint()


# f = h5py.File("/Users/xyz/Data/rlreloaded_data/results/icml-ram-atari/fancy-pong_RUN00.h5","r")
# agent.from_h5(f["agent_snapshotssnapshots/0499"])
# agent.pprint()

vc = VarCollection(agent.policy_vars())
# for var in vc.vars():
#     var.set_value(var.get_value() + 1.0*np.random.randn(*var.get_value().shape).astype(floatX))

o = TT.matrix("o",dtype='uint8')
a = agent.ponder({"o":o})["a"]
olda = TT.matrix('olda')
cpd = agent.cpd()
kl = cpd.kl(a, olda).mean()
l2 = agent.policy.l2()*1e-5
totloss = kl+l2
losses = TT.stack(totloss, kl, l2)
loss_names = ["total","kl","l2"]
gradloss = TT.grad(totloss, vc.vars())

th = TT.vector("th")
datafilename = fetch_file(args.datafile)
datafile = h5py.File(datafilename,"r")

def to2d(x):
    return x.reshape(x.shape[0],-1)

import itertools
ntraj = args.n_traj or len(datafile)
print "USING %i TRAJECTORIES"%ntraj
if args.obs_mode == "ram":
    odata = np.concatenate([grp['ram'].value for grp in itertools.islice(datafile.itervalues(),ntraj)])
    odata = np.squeeze(odata)
elif args.obs_mode == "image":
    odata = np.concatenate([grp['image'].value for grp in itertools.islice(datafile.itervalues(),ntraj)])
    odata = to2d(np.array([mdp.preproc(img) for img in odata]))
    odata = to2d(np.array([np.roll(odata,i-3,axis=0) for i in xrange(4)]).transpose(1,0,2))
    # odata += np.random.randint(odata.shape[0], low=0,high=3).astype('uint8')
else:
    raise NotImplementedError
oldadata = np.concatenate([grp['a'] for grp in itertools.islice(datafile.itervalues(),ntraj)]).astype(floatX)


permute_inds = np.random.permutation(odata.shape[0])
odata = odata[permute_inds]
oldadata = oldadata[permute_inds]

sh_o = theano.shared(odata,name="o",borrow=True)
sh_olda = theano.shared(oldadata,name="olda")
# datafile.close()
start = TT.lscalar('start')
stop = TT.lscalar('stop')
givens = {o:sh_o[start:stop],olda:sh_olda[start:stop]}

var2slice = {var:thsli for (var,thsli) in zip(vc.vars(),symbolic.unflatten2(th,vc.vars()))}
lossesofth = theano.clone(losses, replace=var2slice)
gradlossofth = symbolic.flatten(theano.clone(gradloss,replace=var2slice))


dth = TT.vector("dth")
subsamp = 20
given_subsamp = {o:sh_o[start:stop:subsamp],olda:sh_olda[start:stop:subsamp]}
dparams = symbolic.unflatten(dth, vc.var_shapes(), vc.vars())

fvp = TT.Lop(a, vc.vars(), cpd.fvp(a,TT.Rop(a, vc.vars(), dparams)))
fvpofth = symbolic.flatten(theano.clone(fvp,replace=var2slice))/o.shape[0]

flosses = theano.function([th,start,stop],lossesofth,givens=givens,allow_input_downcast=True)
fgradloss = theano.function([th,start,stop],gradlossofth,givens=givens,allow_input_downcast=True)
fmetric = theano.function([th,dth,start,stop],fvpofth,givens=given_subsamp,allow_input_downcast=True,on_unused_input='ignore')

n_total = sh_o.get_value(borrow=True).shape[0]
n_train = int(n_total*.75)
n_test = n_total-n_train


hdf = setup_outfile(params,agent)
diags = defaultdict(list)
tstart = time.time()

agent.pprint()



def diagnostics_update(th): #pylint: disable=W0621
    test_losses = flosses(th,n_train,n_total)
    train_losses = flosses(th,0,n_test) # don't need to do whole training set here
    for (loss,name) in zip(test_losses,loss_names):
        diags["test_"+name].append(loss)
    for (loss,name) in zip(train_losses,loss_names):
        diags["train_"+name].append(loss)
    diags["time"].append(time.time()-tstart)
    print "********* Done %i iterations **********"%(len(diags["time"])-1)
    print tabulate([(name,ts[-1]) for (name,ts) in sorted(diags.items())])

if args.opt_method == "lbfgs":
    th =  vc.var_values_flat()
    iteration = 0
    while True:
        for th in lbfgs(lambda th: flosses(th,0,n_train)[0], lambda th:fgradloss(th,0,n_train),th,maxiter=9999):
            print "***** iteration %i *****"%iteration        
            diagnostics_update(th)
            iteration += 1
            if iteration > args.n_iter: break


        if is_save_iter(iteration, args.snapshot_period, args.n_iter):
            vc.set_var_values_flat(th)
            save_agent_snapshot(hdf, agent, iteration)            

        if iteration > args.n_iter: break
        print "Resetting LBFGS hessian"


# asdf

elif args.opt_method == "sgd":
    batch_size = 50
    batches = [(start,start+batch_size) for start in xrange(0,n_train-(n_train%batch_size),batch_size)]

    th = vc.var_values_flat()
    for (iteration,state) in enumerate(adaptive_descent(
        f = lambda th,(start,stop): flosses(th,start,stop)[0],
        gradf = lambda th,(star,stop): fgradloss(th,start,stop),
        x0 = th,
        batches = batches,
        eval_batch = (0,n_train),
        method='sgd',
        try_vary_every=5, 
        initial_stepsize=1e-3,
        max_iter=200000)):
        print "***** iteration %i *****"%iteration
        th = state.x
        diagnostics_update(th)

        if is_save_iter(iteration, args.snapshot_period, args.n_iter):
            vc.set_var_values_flat(th)
            save_agent_snapshot(hdf, agent, iteration)            


elif args.opt_method == "cg":
    th= vc.var_values_flat()
    for iteration in xrange(args.n_iter):
        timers = Timers()
        th = cg_optimize(th,
            floss=timers.wrap(lambda th: flosses(th,0,n_train)[0],"loss"),
            fgradloss=timers.wrap(lambda th: fgradloss(th,0,n_train),"grad"),
            metric_length=.01,
            substeps=1,
            damping=1e-3,
            fmetric=timers.wrap(lambda th,dth: fmetric(th,dth,0,n_train),"hvp")
        )
        timers.disp("Optimization timings")
        diagnostics_update(th)

        if is_save_iter(iteration, args.snapshot_period, args.n_iter):
            vc.set_var_values_flat(th)
            save_agent_snapshot(hdf, agent, iteration)            


elif args.opt_method == "cg_with_rescaling":
    if args.normalize:
        with Message("scaling first layer"):
            agent.net.slayers[0].update(-odata.mean(axis=0)[None,:].astype(floatX), 1/(1e-4+odata.std(axis=0))[None,:].astype(floatX))

    th= vc.var_values_flat()
    for iteration in xrange(args.n_iter):
        timers = Timers()
        th = cg_optimize(th,
            floss=timers.wrap(lambda th: flosses(th,0,n_train)[0],"loss"),
            fgradloss=timers.wrap(lambda th: fgradloss(th,0,n_train),"grad"),
            metric_length=.01,
            substeps=1,
            damping=1e-3,
            fmetric=timers.wrap(lambda th,dth: fmetric(th,dth,0,n_train),"hvp")
        )
        timers.disp("Optimization timings")
        diagnostics_update(th)

        if args.renormalize:
            vc.set_var_values_flat(th)
            with Message("renormalizing"):
                agent.net.renormalize(odata)            
            th = vc.var_values_flat()


        if is_save_iter(iteration, args.snapshot_period, args.n_iter):
            vc.set_var_values_flat(th)
            save_agent_snapshot(hdf, agent, iteration)            



elif args.opt_method == "none":
    print "not optimizing"
    sys.exit(0)

else:
    raise NotImplementedError


agent.policy.set_var_values_flat(th)
with Message("computing cost"):
    from control4.core.rollout import rollout
    from control4.core.ml_agent import MLAgent
    with Message("stochastic agent"):
        cs = [np.sum(rollout(mdp, agent, 999999, save_arrs=("c"))[1]['c']) for _ in xrange(3)]
        print "mean episode cost",np.mean(cs)
    with Message("ml agent"):
        mlagent = MLAgent(agent)
        cs = [np.sum(rollout(mdp, mlagent, 999999, save_arrs=("c"))[1]['c']) for _ in xrange(3)]
        print "mean episode cost",np.mean(cs)        
write_diagnostics(hdf,diags)
hdf["runinfo"]["done"]=True

