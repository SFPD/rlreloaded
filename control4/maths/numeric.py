import numpy as np

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def uniform(lo,hi):
    return np.random.rand(lo.size)*(hi-lo) + lo

def flatten(arrs):
    return np.concatenate([arr.flatten() for arr in arrs])

def unflatten(flatarr, shapes):    
    arrs = []
    n = 0        
    for shape in shapes:
        size = np.prod(list(shape))
        arrs.append( flatarr[n:n+size].reshape(shape) )
        n += size
    return arrs

def repeat_row(x,n):
    return np.repeat(x.reshape(1,-1),n,axis=0)

def interp2d(x,xp,yp):
    """Same as np.interp, but yp is 2d"""
    yp = np.asarray(yp)
    assert yp.ndim == 2
    return np.array([np.interp(x,xp,col) for col in yp.T]).T

def normr(x):
    assert x.ndim == 2
    return x/norms(x,1)[:,None]

def normc(x):
    assert x.ndim == 2
    return x/norms(x,0)[None,:]

def norms(x,ax):
    return np.sqrt(np.square(x).sum(axis=ax))

def intround(x):
    return np.round(x).astype('int32')

def deriv(x):
    T = len(x)
    return interp2d(np.arange(T),np.arange(.5,T-.5),x[1:]-x[:-1])

def linspace2d(start,end,n):
    cols = [np.linspace(s,e,n) for (s,e) in zip(start,end)]
    return np.array(cols).T

def remove_duplicate_rows(mat):
    diffs = mat[1:] - mat[:-1]
    return mat[np.r_[True,(abs(diffs) >= 1e-5).any(axis=1)]]

def loglinspace(a,b,n):
    return np.exp(np.linspace(np.log(a),np.log(b),n))

def vector_angle(a,b):
    return np.arccos(a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b))

def cross_corr_1d(ypred,y):
    assert y.ndim == 1 and ypred.ndim == 1
    return np.corrcoef(ypred,y)[0,1]    

def explained_variance_1d(ypred,y):
    assert y.ndim == 1 and ypred.ndim == 1    
    
    def demean(a): return a - a.mean()
    return 1 - (demean(y-ypred)**2).mean()/(demean(y)**2).mean()

def cross_corr(ypred,y):
    if ypred.ndim == 1:
        return cross_corr_1d(ypred,y)
    if ypred.ndim == 2:
        return [cross_corr_1d(ypred1,y1) for (ypred1, y1) in zip(ypred.T, y.T)]

def explained_variance(ypred,y):
    if ypred.ndim == 1:
        return explained_variance_1d(ypred,y)
    if ypred.ndim == 2:
        return np.array([explained_variance_1d(ypred1,y1) for (ypred1, y1) in zip(ypred.T, y.T)])

def vec_cosine(x,y):
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))


def scale_pm1(x,lohi):
    lo = lohi[0]
    hi = lohi[1]
    mid = (lo+hi)/2
    halfrange = (hi-lo)/2
    return (x - mid[None,:])/halfrange[None,:]

def unscale_pm1(sx,lohi):
    lo = lohi[0]
    hi = lohi[1]
    mid = (lo+hi)/2
    halfrange = (hi-lo)/2
    return sx*halfrange[None,:] + mid[None,:]
