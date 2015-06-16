from control4.maths.symbolic import apply_nonlinearity
from control4.maths.numeric import normc
from control4.config import floatX
import theano,theano.tensor as TT
import numpy as np
from control4.misc.collection_utils import concatenate

class ParameterizedFunc(object):
    """
    Function with theano shared variables as parameters
    """

    def __call__(self,*inputs):
        raise NotImplementedError

    def opt_vars(self):
        raise NotImplementedError

    def extra_vars(self):
        return []

class DenseLayer(ParameterizedFunc):
    """
    Affine transformation + nonlinearity
    """


    def __init__(self, src_sizes, targ_size, nonlinearity, src_names=None, targ_name=None, col_norm=1.0):
        """
        src_sizes: dimensionality of inputs
        targ_size: dimensionality of output
        nonlinearity: str, see apply_nonlinearity
        src_names: optional, list of str names of inputs to this layer
        targ_name: optional, str name of output of this layer
        """

        if src_names is None: src_names = ["unnamedinput%i"%i for i in xrange(len(src_sizes))]
        if targ_name is None: targ_name = "unnamedoutput"

        n_in = len(src_sizes)
        Ws_init = [normc(randn_init(src_size,targ_size))*(col_norm/np.sqrt(n_in)) for src_size in src_sizes]
        b_init = np.zeros((1,targ_size),floatX)

        self.Ws = [theano.shared(W,name="W_%s_%s"%(src_name,targ_name)) for (W,src_name) in zip(Ws_init,src_names)]
        self.b = theano.shared(b_init,name="b_%s"%targ_name)
        self.b.type.broadcastable = (True,False)
        self.nonlinearity = nonlinearity

    def __call__(self,*inputs):
        assert len(inputs)==len(self.Ws)
        summands = []
        summands.extend([X.dot(W) for (W,X) in zip(self.Ws,inputs)])
        summands.append(self.b)
        return apply_nonlinearity(TT.add(*summands),self.nonlinearity)
        
    def opt_vars(self):
        return self.Ws + [self.b]

    def extra_vars(self):
        return []

class ElemwiseLinearLayer(ParameterizedFunc):

    def __init__(self, size,name_prefix=""):
        self.trans = theano.shared(np.zeros((1,size),floatX),name=name_prefix+'trans')
        self.scaling = theano.shared(np.ones((1,size),floatX),name=name_prefix+'scaling')
        self.trans.type.broadcastable = (True,False)
        self.scaling.type.broadcastable = (True,False)

    def __call__(self,X):
        if X.dtype != floatX: X = TT.cast(X,floatX)
        return (X+self.trans)*self.scaling

    def opt_vars(self):
        return []
        
    def extra_vars(self):
        return [self.trans,self.scaling]

    def update(self, newtrans, newscaling):
        self.trans.set_value(newtrans.reshape(1,-1))
        self.scaling.set_value(newscaling.reshape(1,-1))

    def update_with_compensation(self,newtrans,newscaling, nextW, nextb):
        newtrans = newtrans.reshape(1,-1)
        newscaling = newscaling.reshape(1,-1)
        trans_old = self.trans.get_value()
        scaling_old = self.scaling.get_value()
        self.trans.set_value(newtrans)
        self.scaling.set_value(newscaling)

        nextW_old = nextW.get_value(borrow=True)
        nextW_new = (scaling_old/newscaling).T * nextW_old
        nextW.set_value(nextW_new)
        nextb_old = nextb.get_value(borrow=True)
        nextb_new = nextb_old + (trans_old * scaling_old).dot(nextW_old) - (newtrans * newscaling).dot(nextW_new)
        nextb.set_value(nextb_new.reshape(1,-1))

class MLP(ParameterizedFunc):
    """
    A sequence of DenseLayer
    """
    def __init__(self, sizes, nonlinearities, names=None, init_col_norms=None):
        """
        sizes: number of units at each layer
        i.e., we have (sizes-1) weight matrices and nonlinearities
        """
        assert len(nonlinearities) == len(sizes)-1
        if names is None: names = [str(i) for i in xrange(len(sizes))]
        else: assert len(names) == len(sizes)
        if init_col_norms is None: init_col_norms = [1.0 for _ in xrange(len(nonlinearities))]
        else: assert len(init_col_norms) == len(nonlinearities)
        self.layers = []
        prev_output_size = sizes[0]
        prev_name = names[0]
        for (output_size,nonlinearity,name,col_norm) in zip(sizes[1:],nonlinearities,names[1:],init_col_norms):
            layer = DenseLayer([prev_output_size],output_size,nonlinearity=nonlinearity,src_names=[prev_name],targ_name=name,col_norm=col_norm)
            self.layers.append(layer)
            prev_output_size=output_size
            prev_name = name

    def __call__(self,X):
        for layer in self.layers:
            X = layer(X)
        return X

    def opt_vars(self):
        out = []
        for layer in self.layers:
            out.extend(layer.opt_vars())
        return out

    def extra_vars(self):
        return []



class LayerChain(ParameterizedFunc):
    """
    A chain of ParameterizedFunc with one input and one output
    """
    def __init__(self,layers):
        self.layers = layers

    def __call__(self,X):
        for layer in self.layers:
            X = layer(X)
        return X

    def opt_vars(self):
        return concatenate(layer.opt_vars() for layer in self.layers)

    def extra_vars(self):
        return concatenate(layer.extra_vars() for layer in self.layers)


class NetworkFromFunc(ParameterizedFunc):
    """
    Create a new ParameterizedFunc out of a bunch of smaller ones.
    Pass in a function or lambda to constructor.
    """

    def __init__(self, layers, f):
        self.layers = layers
        self.f = f

    def __call__(self,*inputs):
        return self.f(*inputs)

    def opt_vars(self):
        return concatenate(layer.opt_vars() for layer in self.layers)

    def extra_vars(self):
        return concatenate(layer.extra_vars() for layer in self.layers)


def randn_init(*shape):
    """
    randn with normalized columns
    """
    x = np.random.randn(*shape)
    x /= np.sqrt(np.square(x).sum(axis=0))
    x = x.astype(floatX)
    return x

