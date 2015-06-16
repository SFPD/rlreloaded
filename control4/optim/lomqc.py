"""
Dual gradient descent
"""
import numpy as np
import scipy.optimize as opt

def line_search_armijo(f,xk,pk,gfk,old_fval,c1=.5,min_alpha=1e-10):
    alpha = 1

    gdotp = pk.dot(gfk)

    if VERBOSE: print "%10s %10s"%("alpha","armijo")
    while True:
        if alpha < min_alpha:
            print "WARNING: couldn't satisfy armijo condition. just returning %g"%min_alpha
            return min_alpha

        armijo = (f(xk + alpha*pk) - old_fval)/(gdotp*alpha)
        if VERBOSE: print "%10.3g %10.3g"%(alpha,armijo)

        if armijo > c1:
            return alpha

        alpha *= .5


def lomqc(g, As, eps):
    """
    linear objective multiple quadratic constraints

    minimize g'x
    subject to x'Ax/2 < eps
    forall A
    """
    As = np.array(As)
    def dual(lam,return_xopt=False):
        """
        return lb, subgrad
        """
        A = np.tensordot(As,lam,axes=[[0],[0]])
        xopt = -np.linalg.solve(A,g)
        if return_xopt:
            return xopt
        else:
            viols = .5*np.tensordot(np.tensordot(As,xopt,axes=[[2],[0]]),xopt,axes=[[1],[0]]) - eps
            val = g.dot(xopt) + lam.dot(viols)
            return -val, -viols



    ncnt = len(As)
    lam0 = np.ones(ncnt)*1e-1

    # grad check:
    # import numdifftools as ndt
    # g1 = ndt.Gradient(lambda y: dual(y)[0])(lam0)    
    # g2 = dual(lam0)[1]
    # assert np.allclose(g1,g2)

    lam_opt, _val_opt, _d = opt.fmin_l_bfgs_b(dual, lam0, None,disp=0,bounds = [(1e-9,None) for _ in xrange(ncnt)])
    return dual(lam_opt,return_xopt=True)

if __name__ == "__main__":
    N = 4
    K = 7
    g = np.random.randn(N)
    def rand_pd(n):
        x = np.random.randn(n,n)
        x = x.T.dot(x)
        return x
    As = np.array([rand_pd(N) for _ in xrange(K)])
    lomqc(g,As,0.1)

