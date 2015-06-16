import numpy as np, theano.tensor as TT
import theano
from control4.maths.symbolic import floatX,join_columns,lmatrix,fmatrix,is_symbolic_tensor
import cycontrol
import scipy.stats

class CPD(object):
    """
    Conditional Probability Distribution
    """

    def mls(self, a_na):
        """
        Most likely sample 
        """
        raise NotImplementedError

    def liks(self, a_na, b_nb):
        """
        p(b | a)
        """
        raise NotImplementedError

    def logliks(self, a_na, b_nb):
        """
        log p(b | a)
        """
        raise NotImplementedError

    def draw(self, a_na):
        """
        b ~ p(b | a)
        """
        raise NotImplementedError

    def draw_numeric(self, a_na):
        """
        See draw
        """
        raise NotImplementedError

    def draw_frozen(self, a_na, r_nr):
        """
        b = f(a, r) so that b has distribution
        b ~ p(b | a)
        """
        raise NotImplementedError

    def draw_frozen_numeric(self, a_na, r_nr):
        """
        See draw_frozen
        """
        raise NotImplementedError

    def entropy(self, a_na):
        """
        entropy or differential entropy
        """
        raise NotImplementedError

    def kl(self, a0_na, a1_n):
        """
        KL[ p(b | a1) || p(b | a2)  ]
        """
        raise NotImplementedError

    def a_dim(self):
        raise NotImplementedError

    def b_dim(self):
        raise NotImplementedError

    def r_dim(self):
        raise NotImplementedError

    def fvp(self, a_na, v_na):
        """
        v -> Fv, where F is Fisher information matrix
        """
        raise NotImplementedError

    ################################

    def draw_multi(self, a_na, k):
        """
        Draw k samples (numerically) from p(b | a)
        """
        return np.array([self.draw_numeric(a_na) for _ in xrange(k)]).transpose(1,0,2)


class FactoredCategoricalDistribution(CPD):
    """
    """
    def __init__(self, li_ncat):
        self.li_ncat = li_ncat
        self.slice_starts_ends = compute_slice_starts_ends(li_ncat)

    def mls(self, a_na):
        return join_columns([a_na[:,start:end].argmax(axis=1) for (start,end) in self.slice_starts_ends])

    def liks(self, a_na, b_nb):
        return self._get_factor_entries(a_na, b_nb).prod(axis=1)
    
    def logliks(self, a_na, b_nb):
        loga_na = TT.log(a_na) #pylint: disable=E1111
        return self._get_factor_entries(loga_na, b_nb).sum(axis=1)

    def draw_numeric(self, a_na):
        N = a_na.shape[0]
        R = self.r_dim()
        r_nr = np.random.rand(N,R).astype(floatX)
        return self.draw_frozen_numeric(a_na, r_nr)

    def draw(self, a_na):
        return theano.as_op(itypes=[fmatrix],otypes=lmatrix)(self.draw_numeric)(a_na)

    def draw_frozen_numeric(self, a_na, r_nr):
        return np.array([
            cycontrol.categorical2(a_na[:,start:stop], r_n)
            for ((start,stop),r_n) in zip(self.slice_starts_ends, r_nr.T)
        ]).T
        
    def draw_frozen(self,a_na,r_nr):
        return theano.as_op(itypes=[fmatrix,fmatrix],otypes=lmatrix)(self.draw_frozen_numeric)(a_na,r_nr)

    def a_dim(self):
        return self.slice_starts_ends[-1][-1]

    def b_dim(self):
        return len(self.slice_starts_ends)

    def r_dim(self):
        return len(self.slice_starts_ends)

    def entropy(self,a_na):
        MOD = TT if is_symbolic_tensor(a_na) else np
        li_a_na = [a_na[:,start:end] for (start, end) in self.slice_starts_ends]
        return MOD.sum([categorical_entropy(p_nk) for p_nk in li_a_na],axis=0)

    def kl(self,a0_na,a1_na):
        MOD = TT if is_symbolic_tensor(a0_na) else np
        li_a0sli= [a0_na[:,start:end] for (start, end) in self.slice_starts_ends]
        li_a1sli = [a1_na[:,start:end] for (start, end) in self.slice_starts_ends]
        return MOD.sum([categorical_kl(a0sli, a1sli) for (a0sli,a1sli) in zip(li_a0sli, li_a1sli)],axis=0)

    def fvp(self, a_na, v_na):
        return v_na/a_na

    ################################
        
    def _get_factor_entries(self, array_na, factoridx_nb):
        startinds_b = np.array([start for (start,_) in self.slice_starts_ends])
        N = array_na.shape[0]
        B = self.b_dim()
        MOD = TT if is_symbolic_tensor(array_na) else np
        nidx_nb = MOD.arange(N)[:,None] + MOD.zeros([B],dtype='int64')
        aidx_nb = factoridx_nb + startinds_b[None,:]
        return array_na[nidx_nb, aidx_nb]


class DiagonalGaussian(CPD):
    """
    Gaussian with diagonal covariance.
    Distribution is described by mean concatenated with standard deviations
    """
    def __init__(self,d):
        self.d = d
        CPD.__init__(self)

    def mls(self, a_na):
        return a_na[:, :self.d]

    def liks(self,a_na, b_nb):
        mu_nd = a_na[:, :self.d]
        sig_nd = a_na[:, self.d:]
        prodsig_n = TT.prod(sig_nd,axis=1)
        out = TT.exp( TT.square((mu_nd - b_nb)/sig_nd).sum(axis=1) * -.5 ) / (np.cast[floatX](np.sqrt(2*np.pi)**self.d) * prodsig_n)
        assert out.dtype==floatX
        return out

    def logliks(self,a_na, b_nb):
        mu_nd = a_na[:, :self.d]
        sig_nd = a_na[:, self.d:]
        sumlogsig_n = TT.sum(TT.log(sig_nd),axis=1)
        out = TT.square((mu_nd - b_nb)/sig_nd).sum(axis=1) * -.5 -  sumlogsig_n - self.d * np.log(np.sqrt(2*np.pi))
        out = TT.cast(out,floatX)
        return out

    def draw_numeric(self, a_na):
        mu_nd = a_na[:, :self.d]
        sig_nd = a_na[:, self.d:]
        return np.random.randn(a_na.shape[0],self.d).astype(floatX)*sig_nd + mu_nd

    def draw(self, a_na):
        return theano.as_op(itypes=[fmatrix],otypes=fmatrix)(self.draw_numeric)(a_na)

    def draw_frozen_numeric(self, a_na, r_nr):
        randn_nr = scipy.stats.norm.ppf(r_nr).astype(floatX)
        mu_nd = a_na[:, :self.d]
        sig_nd = a_na[:, self.d:]
        return randn_nr*sig_nd + mu_nd

    def draw_frozen(self, a_na, r_nr):
        return theano.as_op(itypes=[fmatrix,fmatrix],otypes=fmatrix)(self.draw_frozen_numeric)(a_na,r_nr)

    def a_dim(self):
        return self.d*2

    def b_dim(self):
        return self.d

    def r_dim(self):
        return self.d

    def entropy(self,ms_nk):
        MOD = TT if is_symbolic_tensor(ms_nk) else np
        sig_nd = ms_nk[:,self.d:]
        return (MOD.log(sig_nd) + .5*np.log(2*np.pi*np.e)).sum(axis=1)

    def kl(self,ms1_nk, ms2_nk):
        MOD = TT if is_symbolic_tensor(ms1_nk) else np
        mu1_nd = ms1_nk[:,:self.d]
        sig1_nd = ms1_nk[:,self.d:]
        mu2_nd = ms2_nk[:,:self.d]
        sig2_nd = ms2_nk[:,self.d:]
        return (MOD.log(sig2_nd/sig1_nd) + (MOD.square(sig1_nd) + MOD.square(mu1_nd - mu2_nd))/(2*MOD.square(sig2_nd)) - .5).sum(axis=1)

    def fvp(self, ms_nk, v_nk):
        sig_nd = ms_nk[:,self.d:]
        prec_nd = 1.0/TT.square(sig_nd)
        return TT.concatenate([prec_nd, 2.0*prec_nd],axis=1)*v_nk


def categorical_entropy(p_nk):
    MOD = TT if is_symbolic_tensor(p_nk) else np
    return -(p_nk * MOD.log(p_nk)).sum(axis=1)
def categorical_kl(p1_nk, p2_nk):
    MOD = TT if is_symbolic_tensor(p1_nk) else np
    return (p1_nk * MOD.log(p1_nk / p2_nk)).sum(axis=1)


################################ 

def compute_slice_starts_ends(sizes):
    """
    Given that ``sizes`` gives the number of categories, this gives the starts and ends
    of the distribution parameter
    """
    curstart = 0
    out = []
    for sz in sizes:
        out.append( (curstart, curstart+sz) )
        curstart += sz
    return out


def test():

    def frandn(*shp):
        return np.random.randn(*shp).astype(floatX)
    def rowsumto1(arr):
        return arr / arr.sum(axis=1)[:,None]


    test_cases = [
        {
            "cpd" : DiagonalGaussian(2),
            "a" : np.concatenate([frandn(3,2),frandn(3,2).__abs__()],axis=1)
        },
        {
            "cpd" : FactoredCategoricalDistribution([2,3]),
            "a" : np.concatenate([rowsumto1(frandn(4,2)),rowsumto1(frandn(4,3))],axis=1)
        },
    ]
    
    theano.config.compute_test_value = 'raise'

    for case in test_cases:
        cpd = case["cpd"]
        a = case["a"]
        v = np.random.randn(*a.shape).astype(floatX)
    
        sa = TT.matrix("a")
        sa.tag.test_value = a
        sa1 = TT.matrix("a1")
        sa1.tag.test_value = a
        sv = TT.matrix("v")
        sv.tag.test_value = v
        fvp1 = cpd.fvp(sa,sv)
        kl = cpd.kl(sa,sa1).sum()
        fvp2 = theano.clone(TT.grad((TT.grad(kl,sa)*sv).sum(),sa),replace={sa1:sa}) #pylint: disable=E1103
        # f_fvp1 = theano.function([sa,sv],fvp1)
        # f_fvp2 = theano.function([sa,sv],fvp2)
        # assert np.allclose(f_fvp1(a,v),f_fvp2(a,v))
        assert np.allclose(fvp1.tag.test_value, fvp2.tag.test_value) #pylint: disable=E1103
        assert np.allclose(kl.tag.test_value, 0)

        sb = cpd.draw(sa)
        lik = cpd.liks(sa,sb)
        loglik = cpd.logliks(sa,sb)
        loglik2 = TT.log(lik) #pylint: disable=E1111
        assert np.allclose(loglik.tag.test_value, loglik2.tag.test_value)

        assert a.shape[1] == cpd.a_dim()
        assert sb.tag.test_value.shape[1] == cpd.b_dim() #pylint: disable=E1103

        print "case passed"



if __name__ == "__main__":
    test()
