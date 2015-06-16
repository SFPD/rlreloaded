import cma
import numpy as np

def cma_gen(f,th,batch_size,n_iter,sigma):
    """
    Wraps up optimizer from CMA package so it has standard generator interface
    """
    optim = cma.CMAEvolutionStrategy(th,sigma)
    for _ in xrange(n_iter): 
        X,fit = optim.ask_and_eval(f,number=batch_size)
        optim.tell(X,fit)
        result = optim.result()
        yield {"ys":fit,"ymean":np.mean(fit),"th":result[0]}

