import cPickle,gzip
import theano.tensor as TT,theano #pylint: disable=F0401

from control4.misc.var_collection import VarCollection
from control4.maths import symbolic
import numpy as np
from collections import defaultdict
from control4.optim.adaptive_descent import adaptive_descent
import time
from tabulate import tabulate
from control4.config import floatX,setup_logging
from control4.nn.mlp_with_rescaling import MLPWithRescaling

if __name__ == "__main__":
    np.random.seed(1)
    setup_logging()

    f = gzip.open("/Users/xyz/Downloads/mnist.pkl.gz", 'rb')
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest) = cPickle.load(f)

    n_classes = 10
    net = MLPWithRescaling([Xtest.shape[1],500,n_classes],["tanh","softmax"] )
    f = gzip.open("/Users/xyz/Downloads/mnist.pkl.gz", 'rb')
    (Xtrain,ytrain), (Xvalid,yvalid), (Xtest,ytest) = cPickle.load(f)

    X = TT.matrix("X")
    start = TT.lscalar('start')
    stop = TT.lscalar('stop')
    N = stop-start
    th = TT.vector('th')
    ytrue = TT.lvector("y")
    ypred = net(X)
    optvars = VarCollection(net.opt_vars())
    optvars.disp()
    VarCollection(net.other_vars()).disp()
    l2 = optvars.l2()*1e-3
    err = TT.neq(ypred.argmax(axis=1),ytrue).mean(axis=0)
    loss = TT.nnet.categorical_crossentropy(ypred,ytrue).mean() + l2
    replace_dict = {v:thslice for (v,thslice) in zip(optvars.vars(),symbolic.unflatten(th,optvars.var_shapes(), optvars.vars()))}

    Xall = theano.shared(np.concatenate([Xtrain,Xvalid,Xtest],axis=0)*(np.random.rand(1,Xtrain.shape[1])*99999+1).astype(floatX),name='Xall')
    yall = theano.shared(np.concatenate([ytrain,yvalid,ytest],axis=0),name='yall')

    flosses = theano.function([th,start,stop],theano.clone(TT.stack(loss,err,l2),replace=replace_dict),givens={X:Xall[start:stop],ytrue:yall[start:stop]})
    fgradloss = theano.function([th,start,stop],theano.clone(symbolic.flatten(TT.grad(loss,optvars.vars())),replace=replace_dict),givens={X:Xall[start:stop],ytrue:yall[start:stop]})

    means,stds = net.activation_stats(X)
    fstats = theano.function([start,stop],theano.clone(means+stds),givens={X:Xall[start:stop]})

    th0 = optvars.var_values_flat()
    n_train = ytrain.size
    n_valid = yvalid.size
    n_total = n_train + n_valid
    batch_size = 100

    loss_names = ["total","err","l2"]

    n_stats = len(means)

    diags = defaultdict(list)
    def diagnostics_update(th):
        test_losses = flosses(th,n_train,n_total)
        train_losses = flosses(th,0,n_valid)
        for (loss,name) in zip(test_losses,loss_names):
            diags["test_"+name].append(loss)
        for (loss,name) in zip(train_losses,loss_names):
            diags["train_"+name].append(loss)
        diags["time"].append(time.time()-tstart)
        print tabulate([(name,ts[-1]) for (name,ts) in sorted(diags.items())])

    tstart = time.time()

    for state in adaptive_descent(
        lambda th,(start,stop) : flosses(th,start,stop)[0],
        lambda th,(start,stop) : fgradloss(th,start,stop),
        th0,
        [(start,start+batch_size) for start in xrange(0,n_train-(n_train%batch_size),batch_size)],
        (0,n_train),
        initial_stepsize=0.001,
        max_iter=100,
        method="sgd"):
        diagnostics_update(state.x)

        optvars.set_var_values_flat(state.x)
        means_stds = fstats(0,n_valid)
        means = means_stds[0:n_stats]
        stds = means_stds[n_stats:2*n_stats]
        # print "activation stats BEFORE:"
        # print "std(means):",[mean.std() for mean in means]
        # print "mean(stds):",[std.mean() for std in stds]

        # VarCollection(net.other_vars()).disp()

        for (layer,slayer,mean,std) in zip(net.layers,net.slayers,means,stds):
            slayer.update_with_compensation(-mean,1.0/(std+1e-8),layer.Ws[0],layer.b)

        means_stds = fstats(0,n_valid)
        means = means_stds[0:n_stats]
        stds = means_stds[n_stats:2*n_stats]
        # print "activation stats AFTER:"
        # print "std(means):",[mean.std() for mean in means]
        # print "mean(stds):",[std.mean() for std in stds]

        # VarCollection(net.other_vars()).disp()
        state.x = optvars.var_values_flat()