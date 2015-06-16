
from control4.nn.rnn import RNN
from control4.nn.nn import DenseLayer,NetworkFromFunc
from control4.misc.var_collection import VarCollection
from control4.config import floatX,setup_logging,print_theano_config
from control4.algs.alg_params import AlgParams,validate_and_filter_args
from control4.algs.save_load_utils import dump_dict_to_hdf,gen_output_h5_name
from control4.maths import symbolic
import numpy as np
import theano, theano.tensor as TT
import argparse
from tabulate import tabulate
from collections import defaultdict
import time,h5py

class Task(object):
    def __init__(self, input_size, output_size, loss_type):
        self.input_size = input_size
        self.output_size = output_size
        self.loss_type = loss_type

    def generate(self,batch_size,length):
        raise NotImplementedError


class Substitution(Task):

    def __init__(self,seq_length,ab_size):
        self.seq_length=seq_length
        self.ab_size=ab_size
        Task.__init__(self,ab_size+1,ab_size,"ce")

    def generate(self,batch_size):
        Q_tna = np.zeros((self.seq_length,batch_size,self.ab_size+1),floatX)
        A_na = np.zeros((batch_size,self.ab_size),floatX)
        for t in xrange(Q_tna.shape[0]):
            Q_tna[t][np.arange(batch_size),np.random.randint(0,self.ab_size,size=(batch_size,))]=1
        t_n = np.random.randint(0,self.seq_length,size=(batch_size,))
        n_n = np.arange(batch_size)
        _,aidx_n = np.nonzero(Q_tna[t_n,n_n])
        A_na[n_n,aidx_n] = 1
        Q_tna[t_n,n_n,-1] = 1
        A_na[n_n,aidx_n] = 1
        return Q_tna, A_na

class Addition(Task):

    def __init__(self,length):
        self.nin = 2
        self.nout = 1
        self.length=length
        Task.__init__(self,2,1,"se")

    def generate(self, batch_size):
        l = np.random.randint(int(self.length*.1))+self.length
        p0 = np.random.randint(int(l*.1), size=(batch_size,))
        p1 = np.random.randint(int(l*.4), size=(batch_size,)) + int(l*.1)
        data = np.random.uniform(size=(l, batch_size, 2)).astype(floatX)
        data[:,:,0] = 0.
        data[p0, np.arange(batch_size), np.zeros((batch_size,),
                                                      dtype='int32')] = 1.
        data[p1, np.arange(batch_size), np.zeros((batch_size,),
                                                      dtype='int32')] = 1.

        targs = (data[p0, np.arange(batch_size),
                     np.ones((batch_size,), dtype='int32')] + \
                 data[p1, np.arange(batch_size),
                      np.ones((batch_size,), dtype='int32')])/2.
        return data, targs.reshape((-1,1)).astype(floatX)


class TempOrder(Task):
    def __init__(self,length):
        self.length = length
        Task.__init__(self,6,4,"ce")

    def generate(self, batch_size):
        l = self.length
        p0 = np.random.randint(int(l*.1), size=(batch_size,)) + int(l*.1)
        v0 = np.random.randint(2, size=(batch_size,))
        p1 = np.random.randint(int(l*.1), size=(batch_size,)) + int(l*.5)
        v1 = np.random.randint(2, size=(batch_size,))
        targ_vals = v0 + v1*2
        vals  = np.random.randint(4, size=(l, batch_size))+2
        vals[p0, np.arange(batch_size)] = v0
        vals[p1, np.arange(batch_size)] = v1
        data = np.zeros((l, batch_size, 6), dtype=floatX)
        targ = np.zeros((batch_size, 4), dtype=floatX)
        data.reshape((l*batch_size, 6))[np.arange(l*batch_size),
                                    vals.flatten()] = 1.
        targ[np.arange(batch_size), targ_vals] = 1.
        return data, targ


class Multiplication(Task):
    def __init__(self,length):
        self.length=length
        Task.__init__(self,2,1,"se")

    def generate(self, batchsize):
        l = np.random.randint(int(self.length*.1))+self.length
        p0 = np.random.randint(int(l*.1), size=(batchsize,))
        p1 = np.random.randint(int(l*.4), size=(batchsize,)) + int(l*.1)
        data = np.random.uniform(size=(l, batchsize, 2)).astype(floatX)
        data[:,:,0] = 0.
        data[p0, np.arange(batchsize), np.zeros((batchsize,),
                                                      dtype='int32')] = 1.
        data[p1, np.arange(batchsize), np.zeros((batchsize,),
                                                      dtype='int32')] = 1.

        targs = (data[p0, np.arange(batchsize),
                     np.ones((batchsize,), dtype='int32')] * \
                 data[p1, np.arange(batchsize),
                      np.ones((batchsize,), dtype='int32')])
        return data, targs.astype(floatX).reshape((-1,1))


def compute_losses(netout_nk,act_nk,loss_type):
    if loss_type=="ce":
        sP_nk = TT.nnet.softmax(netout_nk) 
        sloss = TT.nnet.categorical_crossentropy(sP_nk,act_nk).mean()#+vc.l2()*.001
        serr = TT.neq(netout_nk.argmax(axis=1),act_nk.argmax(axis=1)).mean()
    elif loss_type=="se":
        sloss = TT.square(netout_nk - act_nk).sum(axis=1).mean()
        serr = sloss
    else:
        raise NotImplementedError
    return (sloss,serr)


class RNNParams(AlgParams):
    mem_size = 20
    cell_type = "gru" # tanh, lstm
    batch_size=50
    opt_method="adaptive_rmsprop" # or sgd
    opt_iters=20
    init_reset_off = 0
    hessian_subsample= 20
    truncate_gradient= -1.0

class TaskParams(AlgParams):
    task = str
    n_examples = 5000
    seq_length = 100
    ab_size = 5    
 
def main():

    setup_logging()
    print_theano_config()
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser(formatter_class=lambda prog : argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=50))
    param_list = [RNNParams,TaskParams]
    for param in param_list: param.add_to_parser(parser)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--outfile",type=str,default="")
    parser.add_argument("--metadata",type=str,default="")

    args = parser.parse_args()
    validate_and_filter_args(param_list, args)
    params = args.__dict__

    mem_size = params["mem_size"]
    n_total = params["n_examples"]
    batch_size = params["batch_size"]
    seq_length = params["seq_length"]
    n_train = int(n_total*.75)
    n_test = n_total-n_train
 
    np.random.seed(params["seed"])

    task_name = params["task"]
    if task_name == "substitution":
        task = Substitution(seq_length,params["ab_size"])
    elif task_name == "addition":
        task = Addition(seq_length)
    elif task_name == "temp_order":
        task = TempOrder(seq_length)
    elif task_name == "multiplication":
        task = Multiplication(seq_length)
    else:
        raise NotImplementedError("Unrecognized task %s"%task_name)

    Q_tna, A_nk = task.generate(n_total)
    Q_tna = theano.shared(Q_tna)
    A_nk = theano.shared(A_nk)

    sQ_tna = TT.tensor3("Q")
    sQ_tna.tag.test_value=Q_tna
    sA_nk = TT.matrix("A")
    sA_nk.tag.test_value=sA_nk
    sN = sA_nk.shape[0] #pylint: disable=E1101
 
    rnn = RNN(mem_size,task.input_size,cell_type=params["cell_type"],truncate_gradient=params["truncate_gradient"]>0 and params["truncate_gradient"])
    final_layer = DenseLayer([mem_size],task.output_size,"none",src_names=["mem"],targ_name="output")
    net = NetworkFromFunc([rnn,final_layer],lambda X,Minit: final_layer(rnn(X,Minit)))

    if args.init_reset_off:
        init_br = rnn.cell.br.get_value(borrow=True)
        init_br -= 1

    sinitm_nm = TT.zeros([sN,mem_size],floatX)
    netout_nk = net(sQ_tna,sinitm_nm)

    sloss,serr = compute_losses(netout_nk,sA_nk,task.loss_type)
    loss_names = ["loss","err"] # loss is smooth function we're optimizing, err is the measure we care about, which might be non-smooth (e.g. 0-1 error)

    optvars = VarCollection(net.opt_vars())
    optvars.disp()
    sgradloss = symbolic.flatten(TT.grad(sloss, optvars.vars()))

    # logp = 
    th = TT.vector('th')
    dth = TT.vector("dth")
    sstart = TT.lscalar("start")
    sstop = TT.lscalar("stop")
    th.tag.test_value = optvars.var_values_flat()
    var_shapes = optvars.var_shapes()
    replace_dict = {v:thslice for (v,thslice) in zip(optvars.vars(),symbolic.unflatten(th,var_shapes, optvars.vars()))}
    givens = {sQ_tna:Q_tna[:,sstart:sstop], sA_nk:A_nk[sstart:sstop]}
    subsamp = params["hessian_subsample"]
    given_subsamp = {sQ_tna:Q_tna[:,sstart:sstop:subsamp], sA_nk:A_nk[sstart:sstop:subsamp]}

    dparams = symbolic.unflatten(dth, optvars.var_shapes(), optvars.vars())
    from control4.core.cpd import FactoredCategoricalDistribution
    cpd = FactoredCategoricalDistribution([task.output_size])
    sfvp = symbolic.flatten(TT.Lop(netout_nk, optvars.vars(), cpd.fvp(netout_nk,TT.Rop(netout_nk, optvars.vars(), dparams))))

    flosses = theano.function([th,sstart,sstop],theano.clone(TT.stack(sloss,serr),replace=replace_dict),givens=givens,allow_input_downcast=True)
    fgradloss = theano.function([th,sstart,sstop],theano.clone(sgradloss,replace=replace_dict),givens=givens,allow_input_downcast=True)
    # meanfvp = Averager(lambda (th,p,sli): sum_count_reducer(tuple(f_fvp(th,p,path.prevm_tg,path.prevj_tb,path.o_tf)) for path in get_local_paths(sli)))
    fmetric = theano.function([th,dth,sstart,sstop],theano.clone(sfvp,replace=replace_dict),givens=givens,allow_input_downcast=True)

    th0 = optvars.var_values_flat()
    diags = defaultdict(list)
    tstart = time.time()

    def diagnostics_update(th):
        test_losses = flosses(th,n_train,n_total)
        train_losses = flosses(th,0,n_test)
        for (loss,name) in zip(test_losses,loss_names):
            diags["test_"+name].append(loss)
        for (loss,name) in zip(train_losses,loss_names):
            diags["train_"+name].append(loss)
        diags["time"].append(time.time()-tstart)
        print tabulate([(name,ts[-1]) for (name,ts) in sorted(diags.items())])

    if params["opt_method"] in ("adaptive_sgd","adaptive_rmsprop"):
        from control4.optim.adaptive_descent import adaptive_descent
        for state in adaptive_descent(
            lambda th,(start,stop) : flosses(th,start,stop)[0],
            lambda th,(start,stop) : fgradloss(th,start,stop),
            th0,
            [(start,start+batch_size) for start in xrange(0,n_train-(n_train%batch_size),batch_size)],
            (0,n_train),
            initial_stepsize=0.1,
            max_iter=params["opt_iters"],
            method=params["opt_method"][len("adaptive_"):]):
            diagnostics_update(state.x)
    elif params["opt_method"] == "lbfgs":
        from control4.optim.lbfgs import lbfgs
        for th in lbfgs(
            lambda th: flosses(th,0,n_train)[0],
            lambda th: fgradloss(th,0,n_train),
            th0,
            maxiter = params["opt_iters"],            
            ):
            diagnostics_update(th)
    elif params["opt_method"] == "cg":
        from control4.optim.cg_optimize import cg_optimize
        th = th0
        for iteration in xrange(params["opt_iters"]):
            th = cg_optimize(th,
                floss=lambda th: flosses(th,0,n_train)[0],
                fgradloss=lambda th: fgradloss(th,0,n_train),
                metric_length=0.1,
                substeps=1,
                damping=1e-3,
                fmetric=lambda th,dth: fmetric(th,dth,0,n_train)
            )
            diagnostics_update(th)

    else:
        raise NotImplementedError("invalid opt method: %s"%params["opt_method"])

    fname = args.outfile or gen_output_h5_name()
    print "saving to",fname
    hdf = h5py.File(fname,"w")
    dump_dict_to_hdf(hdf, "params", args.__dict__)
    hdf.create_group("diagnostics")
    for (diagname, val) in diags.items():
        hdf["diagnostics"][diagname] = val

if __name__ == "__main__":
    main()