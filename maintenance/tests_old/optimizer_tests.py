import numpy as np
import time
from control3.adagrad import adagrad
from control3.sgd import sgd, momentum_sgd
from control3.lbfgs import lbfgs
from control3 import adaptive_descent2
from control3.common_util import chunk_slices2
from control3.common import setup_logging
from control3 import adaptive_descent3
import itertools

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    setup_logging()
    print "*** Creating Logistic Regression Dataset ***"
    l2coeff = 0.01
    def fc(w, X):
        return 1. / (1. + np.exp(-X.dot(w))) 
    def floss(w,X,z):
        c = fc(w,X)
        EPSILON=1e-30
        return -(z*np.log(c+EPSILON) + (1-z)*np.log(1-c+EPSILON)).mean() + 0.5 * l2coeff * w.dot(w)
    def fgradloss(w,X,z):
        c = fc(w,X)
        return ((c-z).reshape(-1,1)* X ).mean(axis=0) + l2coeff * w
    def fhploss(w,s,X,_z):
        c = fc(w,X)
        return ((c * (1-c) * X.dot(s)).reshape(-1,1) * X).mean(axis=0) + l2coeff*s
    def loss_gradient_count(w,X,z):
        c = fc(w,X)
        EPSILON=1e-30
        loss = -(z*np.log(c+EPSILON) + (1-z)*np.log(1-c+EPSILON)).mean() + 0.5 * l2coeff * w.dot(w)
        grad = ((c-z).reshape(-1,1)* X ).mean(axis=0) + l2coeff * w
        count = X.shape[0]
        return loss,grad,count



    np.random.seed(0)
    Ndata = 1000
    Xdim = 10
    wtrue = np.random.randn(Xdim) 
    Xfull = np.random.randn(Ndata,Xdim) * (2.0**np.arange(Xdim))[None,:]
    q = Xfull.dot(wtrue)
    p = 1/(1+np.exp(-q))
    zfull = (np.random.rand(p.size) < p).astype('float64')




    batch_size=10
    n_passes=15
    w0 = np.zeros(wtrue.size)
    f = lambda w,start,end:floss(w,Xfull[start:end],zfull[start:end])
    fgrad = lambda w,start,end:fgradloss(w,Xfull[start:end],zfull[start:end])

    f1 = lambda w,inds:floss(w,Xfull[inds],zfull[inds])
    fgrad1 = lambda w,inds:fgradloss(w,Xfull[inds],zfull[inds])

    fgc1 = lambda w,inds:loss_gradient_count(w,Xfull[inds],zfull[inds])

    
    titlestr = "%10s %10s %10s %10s"
    fmtstr =   "%10s %10s %10.3g %10.3g"

    print titlestr%("alg","pass","time","loss")

    datasource = chunk_slices2(Ndata, batch_size)
    test_data = np.arange(Ndata)
    # tstart = time.time()
    # for (i_pass, w) in enumerate(adaptive_descent.adaptive_sgd(f1, fgrad1, w0, datasource, test_data, initial_search=True)):
    #     t = time.time()-tstart
    #     print fmtstr%("adaptive sgd",i_pass, t, floss(w, Xfull,zfull))   

    # tstart = time.time()
    # for (i_pass, w) in enumerate(adaptive_descent2.adaptive_descent(f1, fgrad1, w0, 1e-3, datasource, test_data, initial_search=True, max_iter=n_passes, method='sgd')):
    #     t = time.time()-tstart
    #     print fmtstr%("adaptive sgd 2",i_pass, t, floss(w, Xfull,zfull))   

    def do_test(opt_name,soln_iterator):
        print "********* TESTING %s **********"%opt_name
        tstart = time.time()
        for (i_pass, w) in enumerate(soln_iterator):
            t = time.time() - tstart
            print fmtstr%(opt_name,i_pass, t, floss(w, Xfull,zfull))   

    # do_test("adaptive_rmsprop", adaptive_descent2.adaptive_descent(f1, fgrad1, w0, datasource, test_data, initial_search=True, max_iter=n_passes, method='rmsprop',max_passes=30))
    do_test("adaptive_rmsprop2", itertools.imap(lambda state:state.x, adaptive_descent3.adaptive_descent(f1, fgrad1, w0,  datasource, test_data, initial_stepsize=.01, initial_search=True, max_iter=n_passes, method='rmsprop',max_passes=30)))
    # do_test("adaptive_rmsprop2_hogwild", itertools.imap(lambda state:state.x, adaptive_descent3.adaptive_descent(f1, fgrad1, w0, 0.008, datasource, test_data, initial_search=True, max_iter=n_passes, method='rmsprop_hogwild',max_passes=30)))

    do_test("adaptive_sgd", adaptive_descent2.adaptive_descent(f1, fgrad1, w0, 0.008, datasource, test_data, initial_search=True, max_iter=n_passes, method='sgd'))
    for armijo_scaling in [1,3,5]:
        do_test("adagrad%i"%armijo_scaling, adagrad(w0, f, fgrad, Ndata, batch_size=batch_size,n_passes=n_passes,armijo_scaling=armijo_scaling,d_init=0))
    for armijo_scaling in [1,3,5]:
        do_test("sgd:%i"%armijo_scaling, sgd(w0, f, fgrad, Ndata, batch_size=batch_size,n_passes=n_passes,armijo_scaling=armijo_scaling))

    do_test("lbfgs",lbfgs(lambda w: floss(w,Xfull,zfull), lambda w: fgradloss(w,Xfull,zfull),w0,maxiter=100,max_corr=25))

    
    # Compute max eigenvalue to determine appropriate parameters for momentum SGD
    from control3.krylov import lanczos2
    fhp = lambda w,dw,start,end: fhploss(w,dw,Xfull[start:end],zfull[start:end])
    g0 = fgrad(w0,0,Ndata)
    Q,H = lanczos2(lambda p: fhp(w0,p,0,Ndata), g0, 15)
    min_ev, max_ev = np.linalg.eigvalsh(H)[[0,-1]]
    alpha = 1.0/Ndata * 4/max_ev
    beta = .99# 1.0 - np.sqrt(min_ev/max_ev)
    #################

    do_test("m-sgd",momentum_sgd(w0, fgrad, Ndata, alpha, beta,n_passes=n_passes, batch_size=batch_size))
