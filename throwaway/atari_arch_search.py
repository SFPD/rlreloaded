from control4.agents.pixel_atari_agent import PixelAtariAgent
from control4.agents.ram_atari_agent import RamAtariAgent
from control4.algs.save_load_utils import get_mdp,setup_outfile,fetch_file
from control4.algs.diagnostics import write_diagnostics
from control4.misc.var_collection import VarCollection
import theano.tensor as TT, theano #pylint: disable=F0401
import numpy as np
from collections import defaultdict
from tabulate import tabulate
import time,json,h5py
from control4.misc.console_utils import Message
from control4.config import setup_logging
from control4.core.rollout import rollout


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datafile")
parser.add_argument("--policy_cfg")
parser.add_argument("--metadata")
parser.add_argument("--seed",type=int)
parser.add_argument("--outfile")
parser.add_argument("--obs_mode",choices=["image","ram"])
args = parser.parse_args()

params = args.__dict__
np.set_printoptions(precision=3)
np.random.seed(args.seed)
setup_logging()

with open(args.policy_cfg,"r") as fh: 
    policy_cfg = json.load(fh)
policy_cfg["is_factored"] = False
vf_cfg = None
mdp = get_mdp("atari:pong",kws={"obs_mode":args.obs_mode})
agent = (PixelAtariAgent if args.obs_mode=="image" else RamAtariAgent)(policy_cfg, vf_cfg, mdp)
agent.pprint()

vc = VarCollection(agent.policy_vars())

o = TT.matrix("o",dtype='uint8')
a = agent.ponder({"o":o})["a"]
b = TT.lmatrix("b")
cpd = agent.cpd()
ce = - cpd.logliks(a, b).mean()
err = TT.neq(cpd.mls(a) , b).mean() #pylint: disable=E1101
l2 = agent.policy.l2()*1e-5
totloss = ce+l2
losses = TT.stack(totloss, ce, l2, err)
loss_names = ["total","ce","l2","01"]
gradloss = TT.grad(totloss, vc.vars())

th = TT.vector("th")
datafilename = fetch_file(args.datafile)
datafile = h5py.File(datafilename,"r")

def to2d(x):
    return x.reshape(x.shape[0],-1)

if args.obs_mode == "ram":
    obs = datafile['ram'].value
else:
    with Message("preprocessing images"):
        obs = datafile['obs'].value
        obs = to2d(np.array([mdp.preproc(img.transpose(1,2,0)) for img in obs]))
        obs = to2d(np.array([np.roll(obs,i-3,axis=0) for i in xrange(4)]).transpose(1,0,2))
bdata = datafile['rewards'].value.argmax(axis=1)[:,None]

permute_inds = np.random.permutation(obs.shape[0])
obs = obs[permute_inds]
bdata = bdata[permute_inds]

sh_o = theano.shared(obs,name="o",borrow=True)
sh_b = theano.shared(bdata,name="b")
# datafile.close()
start = TT.lscalar('start')
stop = TT.lscalar('stop')
givens = {o:sh_o[start:stop],b:sh_b[start:stop]}

from control4.maths import symbolic
var2slice = {var:thsli for (var,thsli) in zip(vc.vars(),symbolic.unflatten2(th,vc.vars()))}
lossesofth = theano.clone(losses, replace=var2slice)
gradlossofth = symbolic.flatten(theano.clone(gradloss,replace=var2slice))

flosses = theano.function([th,start,stop],lossesofth,givens=givens)
fgradloss = theano.function([th,start,stop],gradlossofth,givens=givens)

from control4.optim.adaptive_descent import adaptive_descent

n_total = sh_o.get_value(borrow=True).shape[0]
n_train = int(n_total*.75)
n_test = n_total-n_train

batch_size = 10
batches = [(start,start+batch_size) for start in xrange(0,n_train-(n_train%batch_size),batch_size)]

hdf = setup_outfile(params,agent)
diags = defaultdict(list)
tstart = time.time()

agent.pprint()

for (iteration,state) in enumerate(adaptive_descent(
    f = lambda th,(start,stop): flosses(th,start,stop)[0],
    gradf = lambda th,(star,stop): fgradloss(th,start,stop),
    x0 = vc.var_values_flat(),
    batches = batches,
    eval_batch = (0,n_train),
    method='rmsprop',
    initial_stepsize=1e-3,
    max_iter=200)):
    th = state.x
    print "***** iteration %i *****"%iteration
    test_losses = flosses(th,n_train,n_total)
    train_losses = flosses(th,0,n_test) # don't need to do whole training set here
    for (loss,name) in zip(test_losses,loss_names):
        diags["test_"+name].append(loss)
    for (loss,name) in zip(train_losses,loss_names):
        diags["train_"+name].append(loss)
    diags["time"].append(time.time()-tstart)

    print tabulate([(name,ts[-1]) for (name,ts) in sorted(diags.items())])

write_diagnostics(hdf,diags)
hdf["runinfo"]["done"]=True

