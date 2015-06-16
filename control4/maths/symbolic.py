import theano.tensor as TT
from control4.config import floatX
import numpy as np
TINY = np.finfo(floatX).tiny #pylint: disable=E1101

lmatrix = TT.matrix(dtype='int64').type
fmatrix = TT.matrix(dtype=floatX).type

def exp_and_normalize(x): 
    x = x-x.max(axis=x.ndim-1,keepdims=True)
    q = TT.exp(x) #pylint: disable=E1111
    return q/q.sum(axis=x.ndim-1,keepdims=True)

def pairwise_distance(A,B,metric='sqeuclidean'):
    if metric == "sqeuclidean":
        return (A**2).sum(axis=1)[:,None] + (B**2).sum(axis=1)[None,:] - 2*TT.dot(A,B.T)
    elif metric == "euclidean":
        return TT.sqrt(pairwise_distance(A,B,'sqeuclidean'))
    elif metric == "cityblock":
        return TT.abs_(A[:,None,:] - B[None,:,:]).sum(axis=2)
    else:
        raise NotImplementedError

def hinge(x):
    return TT.maximum(x,0)

def is_symbolic_tensor(arr):
    return isinstance(arr,TT.TensorVariable)

def join_columns(cols):
    assert len(cols) > 0
    return TT.stack(*cols).T #pylint: disable=E1103

def unflatten(flatarr, shapes, symb_arrs):    
    arrs = []
    n = 0        
    for (shape,symb_arr) in zip(shapes,symb_arrs):
        size = np.prod(list(shape))
        arr = flatarr[n:n+size].reshape(shape)
        if arr.type.broadcastable != symb_arr.type.broadcastable:
            arr = TT.patternbroadcast(arr, symb_arr.type.broadcastable)
        arrs.append( arr )
        n += size
    return arrs

def unflatten2(flatarr, symb_arrs):
    shapes = [sarr.get_value(borrow=True).shape for sarr in symb_arrs]
    return unflatten(flatarr, shapes, symb_arrs)

def apply_nonlinearity(x,name):
    if name=="none":
        return x
    elif name=="tanh":
        return TT.tanh(x)
    elif name=="hard_rect":
        return TT.maximum(x,0)
    elif name=="soft_rect":
        return TT.log(1+TT.exp(x))
    elif name=="softmax":
        return TT.nnet.softmax(x)
    elif name=="sigmoid":
        return TT.nnet.sigmoid(x)
    else:
        raise NotImplementedError

def flatten(tensors):
    return TT.concatenate([tensor.flatten() for tensor in tensors])

def rbf(xin, bounds, reses, wraps,normalize=True):
    bounds = np.array(bounds)
    dim = bounds.shape[1]
    assert len(wraps) == dim
    nfeats = np.product(reses)
    prod = None
    for (i_dim, (lo,hi), res,wrap) in zip(xrange(dim), bounds.T, reses, wraps):
        xi = xin[:,i_dim] if xin.ndim == 2 else xin[i_dim]
        b = squared_exp_bumps(xi, lo, hi, res, wrap=wrap, normalize=False)
        if xin.ndim == 1: shufflepat = tuple( (0 if i==i_dim else 'x') for i in xrange(dim))
        else: shufflepat = (0,) + tuple( (1 if i==i_dim else 'x') for i in xrange(dim))
        b = b.dimshuffle(*shufflepat)
        prod = b if (prod is None) else prod*b

    if xin.ndim == 1: a = prod.flatten()    
    else: a = prod.reshape((xin.shape[0], nfeats))
    if normalize:
        an = a/(a.sum(a.ndim-1,keepdims=True)+TINY)
        return an
    else:
        return a

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def squared_exp_bumps(x, lo, hi, n, wrap=False,smoothing=np.float32(0.), normalize=False, sigma_scaling=.75):

    ctrs_shape = (1,)*x.ndim + (n,)

    xbpat = tuple(xrange(x.ndim)) + ('x',)
    xb = x.dimshuffle( *xbpat )

    ctrs = np.linspace(lo, hi, n, endpoint = not wrap).astype(floatX).reshape(ctrs_shape) #pylint: disable=E1103

    sigma = np.float32((hi-lo)/float(n))*sigma_scaling
    # print "sigma: %.2g"%sigma

    wrap_fn = angle_normalize if wrap else lambda x:x

    out = TT.exp( (-.5/(sigma**2 + smoothing**2)).astype(floatX) *  wrap_fn(xb-ctrs)**2 ) #pylint: disable=E1111
    
    if normalize: out = out/(out.sum(axis=out.ndim-1, keepdims=True)+1e-9)
    assert out.dtype == floatX
    return out