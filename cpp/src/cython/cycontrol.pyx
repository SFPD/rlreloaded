# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

INT64 = np.int64
ctypedef np.int64_t INT64_t

FLOAT32 = np.float32
ctypedef np.float32_t FLOAT32_t


ctypedef fused floatx:
    cython.float
    cython.double


def categorical2(np.ndarray[floatx, ndim=2] p_nk, np.ndarray[floatx, ndim=1] z_n):
    cdef int n,k
    cdef float z, acc
    cdef np.ndarray[INT64_t, ndim=1] a_n = np.zeros(p_nk.shape[0],dtype=INT64)
    for n in xrange(p_nk.shape[0]):
        acc=0
        z = z_n[n]
        for k in xrange(p_nk.shape[1]):
            acc += p_nk[n,k]
            if acc > z:
                a_n[n] = k
                break
        else:
            print "WARNING: p_k should sum to 1, z in [0,1]. got p_k=%s, z=%s"%(p_nk[n],z)
            a_n[n]=k//2

    return a_n

def categorical1(np.ndarray[floatx, ndim=1] p_k, float z):
    cdef int k
    cdef float acc=0
    for k in xrange(p_k.shape[0]):
        acc += p_k[k]
        if acc > z:
            return k
    else:
        print "WARNING: p_k should sum to 1. z in [0,1]. got p_k=%s, z=%s"%(p_k,z)
        return k//2

    
def pair_feats(np.ndarray[floatx, ndim=2] x_nk):
    cdef int N,K,n,k1,k2,r
    N = x_nk.shape[0]
    K = x_nk.shape[1]
    cdef np.ndarray[floatx, ndim=2] out = np.zeros((N,K*(K-1)/2),x_nk.dtype)
    for n in xrange(N):
        r=0
        for k1 in xrange(K):
            for k2 in xrange(k1+1,K):
                out[n,r] = x_nk[n,k1] * x_nk[n,k2]
                r += 1

    return out

# def quad_eval(np.ndarray[FLOAT32_t, ndim=1] x, float const, np.ndarray[FLOAT_t, ndim=1] lincoeffs, np.ndarray[FLOAT_t, ndim=1] quadcoeffs):
#     cdef float out = const
#     cdef int i,j,nx
#     nx = x.shape[0]
#     for i in xrange(nx):
#         out += x[i]*lincoeffs[i]
#         for j in xrange(i,nx):
#             out += x[i] * x[j] * quadcoeffs[j]
#     return out

# cdef nck(int n,int k):
#     cdef int prod = 1;
#     cdef int i=0,p;
#     for p in xrange(n,n-k,-1):
#         prod *= p
#     for p in xrange(1,k+1):
#         prod /= p
#     return prod