from __future__ import division
from control4.config import floatX
from control4.optim.krylov import cg
from control4.misc.console_utils import Timers
import scipy.optimize as opt
import scipy,numpy as np
from tabulate import tabulate


def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks = 10,accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    print "fval before",fval
    for (_n_backtracks,stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve/expected_improve
        print "a/e/r", actual_improve, expected_improve, ratio
        if  ratio > accept_ratio and actual_improve > 0:
            print "fval after",newfval
            return True, xnew
    return False, x

def cg_with_ellipsoid_projection(f_Ax, b, f, cg_iters=10,verbose=True,delta=.01):
    """
    Approximately solve the subproblem
    minimize b'x subject to .5*x'*A*x < delta

    Here's how it works:
    CG generates a series of iterates s_1, s_2, ...
    Each step i, rescale the CG iterate, giving rescaled iterate .5*x_i'*A*x_i = delta
    Evaluate f(x_i). If it got worse, then return x_{i-1}
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g %10.3g %10.3g"
    titlestr =  "%10s %10s %10s %10s %10s"
    if verbose: print titlestr%("iteration", "res norm","x norm", "lam est","delta f")

    f0 = f(x)
    fprev = f0
    xscaledprev = x.copy()

    xAx = 0

    for i in xrange(cg_iters):
        z = f_Ax(p)
        pAp = p.dot(z)
        v = rdr / pAp
        x += v*p
        xAx += v**2 * pAp
        r -= v*z
        newrdr = r.dot(r)
        mu = newrdr/rdr
        p = r + mu*p

        rdr=newrdr

        lam = np.sqrt(xAx/(2*delta))
        xscaled = x/lam if i>0 else x.copy()
        fnew = f(xscaled)            
        df = fnew-f0
        if verbose: print fmtstr%(i+1, rdr, np.linalg.norm(x),lam, df)
        if fnew > fprev:
            print "CG: got worse. Rolling back."
            return xscaledprev

        fprev = fnew
        xscaledprev = xscaled

    return x

def test_cg_with_ellipsoid_projection():
    A = np.random.randn(5,5)    
    A = A.T.dot(A)
    b = np.random.randn(5)
    x = cg_with_ellipsoid_projection(lambda x: A.dot(x), b, lambda x:x.dot(A).dot(x), cg_iters=5,verbose=True) #pylint: disable=W0612,W0108
    # assert np.allclose(xAx, x.dot(A).dot(x),rtol=1e-6)



def cg_optimize(th,floss,fgradloss,metric_length, substeps,damping,cg_iters=10, fmetric=None,
    num_diff_eps=1e-4,with_projection=False, use_scipy=False, fancy_damping=0,do_linesearch=True, min_lm = 0.0):
    """
    Use CG to take one or more truncated newton steps, where a line search is used to 
    enforce improvement of the objective.
    """
    step_metric_length = metric_length/substeps**2

    for _istep in xrange(substeps):
        if fancy_damping==0:
            g = fgradloss(th)
        elif fancy_damping==1:
            g,s =fgradloss(th)
            s += 1e-8 # XXX hack
        elif fancy_damping == 2:
            g,s = fgradloss(th)
            s += 1e-8 # XXX hack
            s = np.sqrt(s)

        if fmetric is None:
            f_Hp = lambda p: (fgradloss(th+num_diff_eps*p) - g)/num_diff_eps             #pylint: disable=W0640
        else:
            if fancy_damping:
                f_Hp = lambda p: fmetric(th, p) + damping*(s*p) #pylint: disable=W0640
            else:
                f_Hp = lambda p: fmetric(th, p) + damping*p


        if with_projection:
            fullstep = cg_with_ellipsoid_projection(f_Hp, -g, lambda dth: floss(th + dth), cg_iters=cg_iters,verbose=True)
            th = th + fullstep

        else:
            if use_scipy:
                n = th.shape[0]
                Aop = scipy.sparse.linalg.LinearOperator(shape=(n,n), matvec=f_Hp)
                Aop.dtype = th.dtype
                x, _info = scipy.sparse.linalg.cg(Aop, -g, maxiter=cg_iters)
                stepdir = x
            else:    
                stepdir = cg(f_Hp, -g, verbose=True,cg_iters=cg_iters)

            neggdotstepdir = -g.dot(stepdir)
            if not (neggdotstepdir > 0 and np.isfinite(neggdotstepdir)):
                # Doesn't seem to happen anymore, but I used to see it a lot
                # Maybe due to nondifferentiable stuff like max pooling
                print "CG generated invalid or infinite search direction. Skipping this step."
                damping *= 10
                continue
            shs = .5*stepdir.dot(f_Hp(stepdir))
            lm = np.sqrt(shs / step_metric_length)
            if lm < min_lm:
                print "Clipping lagrange multiplier %8.3f -> %8.3f"%(lm, min_lm)
                lm = min_lm
            print "lagrange multiplier:",lm,"gnorm:",np.linalg.norm(g)
            fullstep = stepdir / lm
            if do_linesearch:
                success,th = linesearch(floss, th, fullstep, neggdotstepdir/lm)
                if not success:
                    print "stopping optimization: couldn't make progress"
                    break
            else:
                th = th + fullstep

    return th


def cg_with_2d_search(th, f_lossdist, f_grad, f_hp, damping, delta, min_over_max_delta=1e-6,reltol=.1,verbose=False):
    timers = Timers()
    with timers.stopwatch("grad"):
        b = - f_grad(th)
    f_Ax = lambda p: f_hp(th, p) + damping*p

    steps = []    
    def callback(dth):
        steps.append(dth.copy())
    with timers.stopwatch("cg"):
        cg(f_Ax, b, callback=callback, cg_iters=15)
    steps = steps[1:]

    with timers.stopwatch("hvps (unecessary)"):
        shss = [.5*dth.dot(f_Ax(dth)) for dth in steps]

    origloss,_ = f_lossdist(th)

    delta2best = {}
    def f(log_stepsize_delta):
        stepsize_delta = np.exp(log_stepsize_delta)
        with timers.stopwatch("compute loss"):
            losses,kls = np.array([f_lossdist(th+(np.sqrt(stepsize_delta / shs)*step).astype(floatX))\
                for (step,shs) in zip(steps, shss)]).T
        losses -= origloss
        if verbose:
            print "stepsize_delta = %g"%stepsize_delta
            print tabulate([range(len(steps)), losses,kls],floatfmt="g")

        scores = np.where(kls < delta,0,1e20*(kls-delta)) + losses
        scores[np.isnan(scores)] = np.inf
        if verbose:
            print "best score",scores.min()

        delta2best[stepsize_delta] = (np.argmin(scores),np.min(scores))
        return scores.min()

    res = opt.minimize_scalar(f, (np.log(delta*min_over_max_delta),np.log(delta)), 
        bounds=(np.log(min_over_max_delta*delta),np.log(delta)),method='bounded',options=dict(xtol=reltol,xatol=reltol))
    sdbest = np.exp(res["x"])
    idxbest,lossbest = delta2best[sdbest]
    print "best stepsize: %g. best loss: %g. best idx: %i."%(sdbest, lossbest, idxbest)
    timers.disp()
    return th+steps[idxbest]*np.sqrt(sdbest/shss[idxbest])



if __name__ == "__main__":
    test_cg_with_ellipsoid_projection()
