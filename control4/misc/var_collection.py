import numpy as np
from control4.config import floatX
import theano.tensor as TT
from control4.misc.h5_utils import setitem_maybe_compressed

class VarCollection(object):
    """
    A collection of theano shared variables
    """
    def __init__(self,vars): #pylint: disable=W0622
        self._vars = vars

    def vars(self):
        return self._vars

    def var_values(self):
        return [var.get_value(borrow=True) for var in self.vars()]

    def var_shapes(self):
        return [par.shape for par in self.var_values()]

    def size(self):
        return sum(np.prod(shape) for shape in self.var_shapes())

    def num_vars(self):
        return len(self.var_shapes())

    def set_var_values(self, parvals):
        for (var, newval) in zip(self.vars(), parvals):
            var.set_value(newval)

    def set_var_values_flat(self, theta):
        theta = theta.astype(floatX)
        arrs = []
        n = 0        
        for shape in self.var_shapes():
            size = np.prod(shape)
            arrs.append(theta[n:n+size].reshape(shape))
            n += size
        assert theta.size == n
        self.set_var_values(arrs)
    
    def var_values_flat(self):
        theta = np.empty(self.size(),dtype=floatX)
        theta += np.nan
        n = 0
        for parval in self.var_values():
            theta[n:n+parval.size] = parval.flat
            n += parval.size
        assert theta.size == n
        return theta

    def to_h5(self,grp):
        for var in self.vars():
            arr = var.get_value(borrow=True)
            setitem_maybe_compressed(grp, var.name, arr)

    def from_h5(self,grp):
        parvals = [grp[var.name].value for var in self.vars()]
        self.set_var_values(parvals)
    
    def l2(self):
        vars = self.vars() #pylint: disable=W0622
        return 0 if len(vars)==0 else TT.add(*(TT.square(par).sum() for par in self.vars()))

    def disp(self):
        from tabulate import tabulate
        print "*************************"
        print tabulate([(par.name, str(parval.shape), parval.mean(), parval.std()) for par in self.vars() for parval in (par.get_value(borrow=True),)],headers=["name","shape","mean","std"])
        print "*************************"

    def disp_unflattened(self,v,stats_func=None,stat_names=None):
        """
        Given a flat vector, e.g. gradient, unpack it into shapes of these parameters
        and print out stats
        """
        from tabulate import tabulate
        from control4.maths.numeric import unflatten
        arrs = unflatten(v,self.var_shapes())
        assert (stats_func is None) == (stat_names is None)
        if stats_func is None:
            stats_func = lambda a: [str(a.shape),a.mean(), a.std(), a.min(), a.max()]
            stat_names = ["shape","mean","std","min","max"]
        rows = []
        for (var,arr) in zip(self.vars(),arrs):
            rows.append([var.name] + stats_func(arr))
        headers = ["var"] + stat_names
        print tabulate(rows, headers=headers)
