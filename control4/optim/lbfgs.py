from __future__ import division
import numpy as np
from collections import deque
import scipy.optimize as opt
import itertools
import logging

class InverseHessianPairs(object):
    """
    LBFGS (inverse) Hessian approximation based on rotating list of pairs (step, delta gradient)
    that are assumed to approximately satisfy secant equation.
    """
    def __init__(self,max_num_pairs):
        self.syrhos = deque([],max_num_pairs) #pylint: disable=E1121
    def add(self,s,y):
        rho = 1./y.dot(s)
        self.syrhos.append((s,y,rho))
        if rho < 0: print "WARNING: rho < 0"
    def mvp(self,g):
        """
        Matrix-vector product        
        Nocedal & Wright Algorithm 7.4
        uses H0 = alpha*I, where alpha = <s,y>/<y,y>
        """
        assert len(self.syrhos) > 0

        q = g.copy()
        alphas = np.empty(len(self.syrhos))
        for (i,(s,y,rho)) in reversed(list(enumerate(self.syrhos))):
            alphas[i] = alpha = rho*s.dot(q)
            q -= alpha*y

        s,y,rho = self.syrhos[-1]
        ydoty = y.dot(y)
        sdoty = s.dot(y)
        gamma = sdoty/ydoty

        r = gamma*q

        for (i,(s,y,rho)) in enumerate(self.syrhos):
            beta = rho * y.dot(r)
            r += s * (alphas[i] - beta)

        return r

    
def lbfgs(f,fgrad,x0,maxiter=100,max_corr=25,grad_norm_tol=1e-9, ihp=None,ls_criteria="armijo"):
    """
    LBFGS algorithm as described by Nocedal & Wright
    In fact it gives numerically identical answers to L-BFGS-B on some test problems.
    """
    x = x0.copy()
    yield x
    if ihp is None: ihp = InverseHessianPairs(max_corr)
    oldg = fgrad(x)
    if ls_criteria=="armijo": fval = f(x)
    p = -oldg/np.linalg.norm(oldg)

    log = logging.getLogger("lbfgs")
    iter_count = 0
    while True:
        # TODO compare line searches
        g=None
        if ls_criteria == "strong_wolfe":
            alpha_star, _, _, fval, _, g = opt.line_search(f,fgrad,x,p,oldg)        
        elif ls_criteria == "armijo":
            import scipy.optimize.linesearch
            alpha_star,_,fval=scipy.optimize.linesearch.line_search_armijo(f,x,p,oldg,fval)
        else:
            raise NotImplementedError

        if alpha_star is None:
            log.error("lbfgs line search failed!")
            break
        s = alpha_star * p
        x += s
        yield x

        iter_count += 1
        
        if iter_count  >= maxiter:
            break

        if g is None: 
            log.debug("line search didn't give us a gradient. calculating")
            g = fgrad(x)

        if np.linalg.norm(g) < grad_norm_tol:
            break


        y = g - oldg
        ihp.add( s,y )
        p = ihp.mvp(-g)
        oldg = g

        log.info("lbfgs iter %i %8.3e",iter_count, fval)

def ipm_lbfgs(f,fgrad,g,gjac,g_hi, x0,maxiter=100,max_corr=25,penalty_decrease_every=20,penalty_decrease_factor=4.0,initial_penalty=1.0):
    """
    Interior point method based on L-BFGS hessian approximation
    """
    penalty = initial_penalty
    def fpen(x):
        fval = f(x)
        gval = g(x)
        # print "fpen",x,fval,gval,fval - penalty * np.log(g_hi - gval).sum()
        if gval >= g_hi:
            return 1e100
        else:
            return fval - penalty * np.log(g_hi - gval).sum()
    def fpengrad(x):
        fgradval = fgrad(x)
        gjacval = gjac(x).reshape(g_hi.size,x.size)
        gval = g(x)        
        assert np.isreal(x).all() and np.isfinite(x).all()
        if gval > g_hi:
            return np.zeros_like(x)
        else:
            return fgradval + (penalty / (g_hi - gval)).dot(gjacval)


    ihp = InverseHessianPairs(max_corr)
    iter_count = 0
    while True:
        # import numdifftools as ndt
        # assert np.allclose(ndt.Gradient(fpen)(x0),fpengrad(x0))
        for x0 in lbfgs(fpen, fpengrad, x0, maxiter=penalty_decrease_every,ihp=ihp):
            iter_count += 1
            yield x0
            if iter_count == maxiter: return
        penalty /= penalty_decrease_factor

def ipm2_lbfgs(flag, fgradlag, x0,maxiter=100,max_corr=25,penalty_decrease_every=20,penalty_decrease_factor=4.0,initial_penalty=1.0):
    penalty = initial_penalty

    ihp = InverseHessianPairs(max_corr)
    iter_count = 0
    while True:
        for x0 in lbfgs(lambda x : flag(x, penalty), lambda x: fgradlag(x, penalty), x0, maxiter=penalty_decrease_every,ihp=ihp):
            iter_count += 1
            yield x0
            if iter_count == maxiter: return
        penalty /= penalty_decrease_factor


def test_constrained_minimization():
    dim_x = 2
    a = np.array([3.0,4.0])
    xsol = np.random.randn(dim_x)
    f = lambda x: a.dot(x) #pylint: disable=W0108
    gradf = lambda x:a
    g = lambda x: np.array([x.dot(x)])
    jacg = lambda x: 2*x.reshape(1,-1)
    g_hi = np.array([100.0])
    for xsol in ipm_lbfgs(f, gradf, g, jacg, g_hi, xsol,maxiter=30,penalty_decrease_every=10):
        pass
    print xsol


def stochastic_lbfgs(w0, f, fgrad, fhp, it_grad_batches, it_steplength, it_hess_batches, M=25, L=20):
    ihp=InverseHessianPairs(M)
    w_I = np.zeros(w0.size)
    w_J = w0
    w = w0.copy()
    armijo_acc = 0
    for (k,data_batch,alpha) in itertools.izip(itertools.count(1),it_grad_batches,it_steplength):
        g = fgrad(w,*data_batch)
        f_before = f(w,*data_batch)
        w_I += w
        if k < 2*L: 
            step = -alpha*g
        else:
            step = -alpha*ihp.mvp(g)
        w += step
        f_after = f(w,*data_batch)
        armijo = (f_before - f_after) / step.dot(-g)
        armijo_acc += armijo
        # assert np.isfinite(armijo) and armijo>0
        if k%L==0:
            w_I /= L            
            s = w_I - w_J
            y = fhp(w_I, s, *it_hess_batches.next())
            assert np.linalg.norm(s) > 0
            ihp.add(s,y)
            w_J = w_I
            w_I = np.zeros(w0.size)
            if 0: print "avg armijo", armijo_acc/L
            armijo_acc=0
    return w


def stochastic_lbfgs2(w0,f,fgrad,fhp,Ndata,n_passes=1,scaling=10.):
    grad_batch_size=100
    hess_batch_size=200

    n_batches = Ndata // grad_batch_size
    def stepsize_iter():
        while True:
            yield scaling/n_batches

    def data_iter(batch_size,n_passes):
        for _i_pass in xrange(n_passes):
            for i_batch in xrange(n_batches):
                # print "batch %i/%i"%(i_batch,n_batches)
                yield (1,i_batch*batch_size,(i_batch+1)*batch_size)

    return stochastic_lbfgs(w0, f, fgrad, fhp, data_iter(grad_batch_size,n_passes), stepsize_iter(), data_iter(hess_batch_size,999))





def test_lbfgs():
    print "*** Testing LBFGS ***"

    np.random.seed(0)
    Q = np.random.randn(100,10)
    H = Q.T.dot(Q)
    b = np.random.randn(10)
    f = lambda x: .5*x.dot(H.dot(x)) + b.dot(x)
    fgrad = lambda x: H.dot(x) + b

    x0 = np.random.randn(10)


    maxiter=5
    soln = opt.minimize(f, x0, method='L-BFGS-B',jac=fgrad,options=dict(maxiter=maxiter))
    x_scipy = soln['x']
    def last(seq):
        elem=None
        for elem in seq: pass
        return elem
    x_my = last(lbfgs(f, fgrad, x0, maxiter=maxiter))
    # assert np.allclose(myx,truex)

    assert np.allclose(x_my, x_scipy)



def test_stochastic_lbfgs():
    print "*** Testing Stochastic LBFGS ***"
    def fc(w, X):
        return 1. / (1. + np.exp(-X.dot(w)))
    def floss(w,X,z):
        c = fc(w,X)
        return -(z*np.log(c) + (1-z)*np.log(1-c)).mean()
    def fgradloss(w,X,z):
        c = fc(w,X)
        return ((c-z).reshape(-1,1)* X ).mean(axis=0)
    def fhp(w,s,X,_z):
        c = fc(w,X)
        return ((c * (1-c) * X.dot(s)).reshape(-1,1) * X).mean(axis=0)

    Ndata = 1000
    Xdim = 10
    wtrue = np.random.randn(Xdim)
    Xfull = np.random.randn(Ndata,Xdim)
    q = Xfull.dot(wtrue)
    p = 1/(1+np.exp(-q))
    zfull = (np.random.rand(p.size) < p).astype('float64')

    w0 = np.zeros(Xdim)
    f = lambda w:floss(w,Xfull,zfull)
    jac=lambda w:fgradloss(w,Xfull,zfull)
    import numdifftools as ndt
    assert np.allclose(ndt.Gradient(f)(w0) , jac(w0),atol=1e-10)
    s = np.random.randn(w0.size)
    assert np.allclose(( jac(w0+1e-6*s)-jac(w0))/1e-6,  fhp(w0,s,Xfull,zfull),atol=1e-7)

    soln = opt.minimize(f, w0, method="BFGS",jac=jac)
    w_bfgs = soln['x']
    assert f(w_bfgs) < f(wtrue)


    soln = opt.minimize(f, w0, method="L-BFGS-B",jac=jac,options=dict(maxcor=25))
    w_lbfgs = soln['x']


    def stepsize_iter():
        while True:
            yield 10./Ndata

    grad_batch_size=1
    hess_batch_size=10
    n_passes=1
    def data_iter(batch_size,n_passes):
        for _i_pass in xrange(n_passes):
            for i_data in xrange(0,Ndata,batch_size):
                yield (Xfull[i_data:i_data+batch_size],zfull[i_data:i_data+batch_size])

    # XXX might want to compute hessian vector product on a bigger minibatch
    w_sl = stochastic_lbfgs(w0, floss, fgradloss, fhp, data_iter(grad_batch_size,n_passes), stepsize_iter(), data_iter(hess_batch_size,999))

    print "initial obj", f(w0)
    print "bfgs",f(w_bfgs)
    print "l-bfgs",f(w_lbfgs)
    print "stochastic-lbfgs",f(w_sl)

if __name__ == "__main__":
    test_constrained_minimization()
    test_lbfgs()
    test_stochastic_lbfgs()



