
from control4.nn.rnn import RNN
from control4.nn.nn import NetworkFromFunc,MLP
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


class RNNParams(AlgParams):
    mem_size = 128
    cell_type = "gru" # tanh, lstm
    batch_size=1
    opt_method="adaptive_rmsprop" # or sgd
    opt_iters=1000

class TaskParams(AlgParams):
    seq_length = 40
 
def make_letter2idx():
    import string #pylint: disable=W0402
    out = {}
    for L in string.letters + ".,;!?':-\" \n@":
        out[L] = len(out)
    return out
LETTER2IDX = make_letter2idx()

if __name__ == "__main__":

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
    batch_size = params["batch_size"]
    seq_length = params["seq_length"]
    shifts = range(1,2)
    n_preds = len(shifts)



    X_tna = TT.tensor3("X") # N blocks of text, each with length T. A is alphabet size
    Y_tnpa = TT.tensor4("Y")
    T = seq_length
    N = X_tna.shape[1]
    ab_size = len(LETTER2IDX)

    with open("/Users/xyz/Downloads/11.txt","r") as fh:
        text = fh.read()
        text = text[text.index("CHAPTER I"):text.index("THE END")]
        defaultidx = LETTER2IDX["@"]
        inds = np.array([LETTER2IDX.get(s,defaultidx) for s in text])
        Xdata = np.zeros((len(inds),ab_size),floatX)
        Xdata[np.arange(len(inds)),inds] = 1.0
        shX = theano.shared(Xdata,name="X")
        n_chunks = (len(Xdata) - n_preds) // seq_length
        # n_chunks = 300 # XXX
        randperm =  np.random.randn(n_chunks).argsort()
        # randperm = np.arange(n_chunks)

    A = ab_size
    n_total = n_chunks
    n_train = int(n_chunks*0.75)
    n_test = n_total - n_train

    np.random.seed(params["seed"])
    char_model = RNN(mem_size,ab_size,cell_type=params["cell_type"],src_name="charin",mem_name="charmem")
    predictor_net = MLP(sizes=[mem_size,n_preds*ab_size],nonlinearities=["none"],names=["charmem","charpred"])

    def gen_prediction(X_tna,Minit_nm):
        M_tnm = char_model(X_tna,Minit_nm,return_all=True)
        u_tn_pa = predictor_net(M_tnm.reshape([seq_length*N,mem_size]))
        u_tnpa = u_tn_pa.reshape([T,N,n_preds,A])
        u_tnpa = u_tnpa - u_tnpa.max(axis=3,keepdims=True)
        prob_tnpa = TT.exp(u_tnpa) #pylint: disable=E1111
        prob_tnpa = prob_tnpa / prob_tnpa.sum(axis=3,keepdims=True)       
        return prob_tnpa 


    net = NetworkFromFunc([char_model,predictor_net],gen_prediction)
    optvars = VarCollection(net.opt_vars())
    optvars.disp()


    Minit_nm = TT.zeros([N,mem_size],floatX)
    prob_tnpa = net(X_tna,Minit_nm)
    prob_tnp_a = prob_tnpa.reshape([T*N*n_preds,A])
    Y_tnp_a = Y_tnpa.reshape([T*N*n_preds,A])
    losses_p = TT.nnet.categorical_crossentropy(prob_tnp_a,Y_tnp_a).reshape([T*N,n_preds]).mean(axis=0)     #pylint: disable=E1101
    l2loss = 1e-5*optvars.l2()
    loss = losses_p.sum() + l2loss
    losses = TT.concatenate([losses_p,[l2loss]])
    gradloss = symbolic.flatten(TT.grad(loss, optvars.vars()))

    # logp = 
    sth = TT.vector('th')
    chunk_inds = TT.lvector("start")
    var_shapes = optvars.var_shapes()
    replace_dict = {v:thslice for (v,thslice) in zip(optvars.vars(),symbolic.unflatten(sth,var_shapes, optvars.vars()))}
    def chunkinds2allinds(ivec):
        return ((ivec*seq_length)[None,:] + TT.arange(seq_length)[:,None]).flatten()

    givens = {}
    allinds = chunkinds2allinds(chunk_inds)
    givens[X_tna] = shX[allinds].reshape([seq_length,chunk_inds.size,ab_size])
    givens[X_tna].type.broadcastable = (False,False,False)

    # givens[Y_tnpa] = givens[X_tna][:,:,None,:]
    givens[Y_tnpa] = TT.concatenate([shX[allinds+k].reshape([seq_length,chunk_inds.size,1,ab_size]) for k in shifts],axis=2)
    givens[Y_tnpa].type.broadcastable=(False,False,False,False)

    flosses = theano.function([sth,chunk_inds],theano.clone(losses,replace=replace_dict),givens=givens,allow_input_downcast=True)
    fgradloss = theano.function([sth,chunk_inds],theano.clone(gradloss,replace=replace_dict),givens=givens,allow_input_downcast=True)

    loss_names = ["%iahead"%i for i in shifts] + ["l2"]

    th0 = optvars.var_values_flat()
    diags = defaultdict(list)
    tstart = time.time()


    def diagnostics_update(th):
        test_losses = flosses(th,randperm[n_train:n_total])
        train_losses = flosses(th,randperm[0:n_test])
        for (loss,name) in zip(test_losses,loss_names):
            diags["test_"+name].append(loss)
        for (loss,name) in zip(train_losses,loss_names):
            diags["train_"+name].append(loss)
        diags["time"].append(time.time()-tstart)
        print tabulate([(name,ts[-1]) for (name,ts) in sorted(diags.items())])

    if params["opt_method"] in ("adaptive_sgd","adaptive_rmsprop"):
        from control4.optim.adaptive_descent import adaptive_descent
        for (iteration,state) in enumerate(adaptive_descent(
            lambda th,inds : flosses(th,inds).sum(),
            lambda th,inds : fgradloss(th,inds), #pylint: disable=W0108
            th0,
            [randperm[start:start+batch_size] for start in xrange(0,n_chunks,batch_size)],
            randperm[np.arange(0,n_train)],
            initial_stepsize=0.05,
            max_iter=params["opt_iters"],
            method=params["opt_method"][len("adaptive_"):])):
            print "ITERATION %i"%iteration
            diagnostics_update(state.x)
        th = state.x
    elif params["opt_method"] == "lbfgs":
        from control4.optim.lbfgs import lbfgs
        for th in lbfgs(
            lambda th: flosses(th,np.arange(0,n_train)).sum(),
            lambda th: fgradloss(th,np.arange(0,n_train)),
            th0,
            maxiter = params["opt_iters"],            
            ):
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


    optvars.set_var_values_flat(state.x)

    def onestep(X_na,M_nm):
        NextM_nm = char_model.cell(M_nm,X_na)
        u_n_pa = predictor_net(NextM_nm)
        u_npa = u_n_pa.reshape([u_n_pa.shape[0],n_preds,A])
        prob_npa = TT.exp(u_npa) #pylint: disable=E1111
        prob_npa = prob_npa / prob_npa.sum(axis=2,keepdims=True)       
        return NextM_nm, prob_npa    
    X_na = TT.matrix("X")
    M_nm = TT.matrix("M")
    fpred = theano.function([M_nm,X_na],onestep(X_na,M_nm),allow_input_downcast=True)
    def idx2onehot(i,n):
        out = np.zeros(n,floatX)
        out[i] = 1
        return out
    from cycontrol import categorical1
    IDX2LETTER = {i:c for (c,i) in LETTER2IDX.iteritems()}
    def onestepahead(m,c):
        x = idx2onehot(LETTER2IDX.get(c,defaultidx),len(LETTER2IDX))
        m,p = fpred(m.reshape(1,-1),x.reshape(1,-1))
        m=m[0]
        p=p[0][0]
        # i = categorical1(p,np.float32(np.random.rand()))
        i = p.argmax()
        c = IDX2LETTER[i]
        return m,c


    import sys
    word = 'Ali'
    m = np.zeros(mem_size)
    for c in word:
        sys.stdout.write(c)
        m,c = onestepahead(m,c)
    sys.stdout.write("|")
    for i in xrange(1000):
        sys.stdout.write(c)
        m,c = onestepahead(m,c)
        # print m[:5]
    print
# if __name__ == "__main__":
#     main()