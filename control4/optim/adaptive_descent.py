from __future__ import division
import logging
import numpy as np
from control4.config import floatX
import ctypes

# XXX should we get rid of state.stepsize

class StochasticOptimizer(object):
    def __init__(self):
        self.count = 0
    def do_pass(self, gradf, state, stepsize,batches):
        raise NotImplementedError
    def initialize_state(self,x,y,stepsize):
        raise NotImplementedError

    def do_pass_and_eval(self, gradf, state, stepsize, batches, f, eval_batch):
        state = self.do_pass(gradf,state,stepsize, batches)
        state.y = f(state.x, eval_batch)
        log = logging.getLogger("adaptive_descent")
        log.info("doing batch with stepsize %8.3e -> %8.3e",stepsize,state.y)
        # print "eval: ",state.y
        self.count += 1
        return state

class SGDState(object):
    def __init__(self,x,y,stepsize):
        self.x = x
        self.y = y
        self.stepsize = stepsize

class SGD(StochasticOptimizer):
    def do_pass(self, gradf, state, stepsize, batches):
        x = state.x.copy()
        for batch in batches:
            g = gradf(x,batch)
            x -= stepsize * g
        return SGDState(x, None, stepsize)
    def initialize_state(self,x,y,stepsize):
        return SGDState(x,y,stepsize)

class RMSPropState(object):
    def __init__(self, x, d, y, stepsize):
        self.x = x
        self.d = d
        self.y = y
        self.stepsize = stepsize

class RMSProp(StochasticOptimizer):
    def do_pass(self, gradf, state, stepsize, batches):
        x = state.x.copy()
        d = state.d.copy()
        for batch in batches:
            g = gradf(x,batch)            
            d += np.square(g)
            d *= .99            
            x -= stepsize/np.sqrt(d) * g

        return RMSPropState(x,d,None,stepsize)

    def initialize_state(self, x, y,stepsize):
        return RMSPropState(x, np.zeros_like(x)+1e-6, y, stepsize)


G_OptimizerInstance = None

def update(batch):
    self = G_OptimizerInstance
    g = self.gradf(self.x, batch)
    self.d += np.square(g)
    with self.lock:
        self.x -= self.stepsize*(g/np.sqrt(self.d))

def set_locals(d):
    G_OptimizerInstance.__dict__.update(d)
def get_d(_):
    return G_OptimizerInstance.d

class RMSPropHogwild(StochasticOptimizer):
    def __init__(self, gradf, n_processes, size):
        StochasticOptimizer.__init__()
        from control4.parallel.parallel import make_pool_maybe_dummy
        import multiprocessing.sharedctypes
        global G_OptimizerInstance #pylint: disable=W0603
        assert G_OptimizerInstance==None
        G_OptimizerInstance = self

        self.gradf = gradf
        self.d = None
        self.stepsize = None
        typ = {'float32':ctypes.c_float, 'float64':ctypes.c_double}[floatX]        
        self.x = np.frombuffer(multiprocessing.sharedctypes.RawArray(typ, np.zeros(size,floatX)),floatX)
        self.lock = multiprocessing.Lock()
        self.pool = make_pool_maybe_dummy(n_processes)

    def do_pass(self, _gradf, state, stepsize, batches):
        # XXX maybe we should set gradf too'
        self.x[:] = state.x
        self.pool.apply(set_locals, dict(d = state.d.copy(), stepsize=stepsize))
        self.pool.map(update, batches)
        d = np.array(self.pool.apply(get_d, None)).mean(axis=0)
        return RMSPropState(self.x.copy(), d, None, stepsize) 

    def initialize_state(self, x, y,stepsize):
        return RMSPropState(x, np.zeros_like(x)+1e-6, y, stepsize)

    def __del__(self):
        global G_OptimizerInstance #pylint: disable=W0603
        G_OptimizerInstance=None


def adaptive_descent(f, gradf, x0, batches, eval_batch, initial_search=True, try_vary_every=5, max_iter=10,method='sgd',
    max_passes=None, initial_state=None, initial_stepsize=None, min_stepsize=1e-10):

    """
    Stochastic optimizer with adaptive stepsize
    The adaptation scheme is basic and foolproof: every once in a while, you run your batch with a larger and smaller stepsize,
    and see which stepsize gives the best final objective value.
    At the very beginning do a simple bracketing search to find the near-optimal initial stepsize.
    """

    if max_passes is None: max_passes = 2*max_iter

    assert (initial_state is None) != (initial_stepsize is None)
    if initial_state is not None: initial_stepsize = initial_state.stepsize

    stepsize_increase_ratio = 5.0
    stepsize_decrease_ratio = 1.0/5.0


    log = logging.getLogger("adaptive_descent")

    if method == "sgd":
        optimizer = SGD()
    elif method == "rmsprop":
        optimizer = RMSProp()
    elif method == "rmsprop_hogwild":
        optimizer = RMSPropHogwild(gradf, -1, x0.size)
    else:
        raise NotImplementedError

    y0 = f(x0, eval_batch)
    if initial_state is None:
        base_state = optimizer.initialize_state(x0, y0, initial_stepsize)
    else:
        base_state = initial_state
    log.info("initial objective: %8.5e",y0)
    yield base_state


    base_state = optimizer.initialize_state(x0,y0,initial_stepsize)

    def better(state, other_state):
        return state.y < other_state.y

    if initial_search:
        log.info("searching for stepsize")        
        state_smallstep = optimizer.do_pass_and_eval(gradf, base_state, base_state.stepsize,batches, f, eval_batch)
        state_bigstep = optimizer.do_pass_and_eval(gradf, base_state, base_state.stepsize * stepsize_increase_ratio,batches, f, eval_batch)
        if better(state_smallstep, state_bigstep):
            state_last = state_smallstep
            state_second_to_last = state_bigstep
            search_stepsize_ratio = stepsize_decrease_ratio
        else:
            state_last = state_bigstep
            state_second_to_last = state_smallstep
            search_stepsize_ratio = stepsize_increase_ratio

        while better(state_last, state_second_to_last):
            if optimizer.count >= max_passes: 
                log.warn("aborting optimization during initial stepsize search because exceed max passes (%i)", optimizer.count)
                return
            state_last, state_second_to_last = optimizer.do_pass_and_eval(gradf, base_state, state_last.stepsize * search_stepsize_ratio,batches, f, eval_batch), state_last
        
        base_state = state_second_to_last
        yield base_state
        log.info("done search. stepsize: %8.5e",base_state.stepsize)


    successful_step_count = 0 # Number of iterations for which we've improved on the first try
    for _ in xrange(max_iter):
        if optimizer.count >= max_passes:
            log.warn("aborting optimization because exceed max passes (%i)", optimizer.count)
            return
        next_state_candidate = optimizer.do_pass_and_eval(gradf, base_state, base_state.stepsize,batches, f, eval_batch)

        if better(next_state_candidate, base_state):
            log.info("step improved objective")
            successful_step_count += 1
            if successful_step_count % try_vary_every == 0:
                log.info("trying to vary stepsize")
                next_state_candidates = [next_state_candidate]
                next_state_candidates.append(optimizer.do_pass_and_eval(gradf, base_state, base_state.stepsize*stepsize_increase_ratio,batches, f, eval_batch))
                next_state_candidates.append(optimizer.do_pass_and_eval(gradf, base_state, base_state.stepsize*stepsize_decrease_ratio,batches, f, eval_batch))
                prev_stepsize = base_state.stepsize
                base_state = min(next_state_candidates, key = lambda state: state.y)
                successful_step_count = 0
                log.info("result: stepsize %8.5e -> %8.5e",prev_stepsize, base_state.stepsize)
            else:
                base_state = next_state_candidate
            yield base_state
        else:
            log.info("step made objective worse (%f > %f)",next_state_candidate.y, base_state.y)            
            successful_step_count = 0
            current_stepsize = base_state.stepsize
            while True:
                if optimizer.count >= max_passes:
                    log.warn("aborting optimization because exceed max passes (%i)", optimizer.count)
                    return
                if current_stepsize < min_stepsize:
                    log.warn("aborting optimization because stepsize too small (%8.5e)", current_stepsize)
                    return                    
                current_stepsize *= stepsize_decrease_ratio
                next_state_candidate = optimizer.do_pass_and_eval(gradf, base_state, current_stepsize,batches, f, eval_batch)
                if better(next_state_candidate, base_state):
                    base_state = next_state_candidate
                    yield base_state
                    break
                else:
                    log.info("shrinking step...")


