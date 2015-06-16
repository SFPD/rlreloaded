
from control4.nn.nn import ParameterizedFunc
from control4.config import floatX
import numpy as np
import theano, theano.tensor as TT
from control4.maths.numeric import normc

def randnf(*shp):
    return np.random.randn(*shp).astype(floatX)

class TanhCell(ParameterizedFunc):
    """
    Basically a DenseLayer with tanh, which has form required by RNN class
    """
    def __init__(self,input_sizes,mem_size,src_names=None,mem_name=None):
        if src_names is None: src_names = ["unnamedinput%i"%i for i in xrange(len(input_sizes))]
        if mem_name is None: mem_name = "mem"
        Wim_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wims = [theano.shared(Wim_val,name="Wim_%s"%src_name) for (Wim_val,src_name) in zip(Wim_vals,src_names)]
        Wmm_val = np.eye(mem_size,dtype=floatX)
        self.Wmm = theano.shared(Wmm_val,name="Wmm_%s"%mem_name)
        bm = np.zeros((1,mem_size),floatX)
        self.bm = theano.shared(bm,broadcastable=(True,False),name="bm_%s"%mem_name)

    def __call__(self,M,*inputs):
        summands = [Xi.dot(Wim) for (Xi,Wim) in zip(inputs,self.Wims)] + [M.dot(self.Wmm),self.bm]
        return TT.tanh(TT.add(*summands))

    def opt_vars(self):
        out = []
        out.extend(self.Wims)
        out.append(self.Wmm)
        out.append(self.bm)
        return out


class GRUCell(ParameterizedFunc):
    """
    Gated Recurrent Unit. E.g., see
    Chung, Junyoung, et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." arXiv preprint arXiv:1412.3555 (2014).
    """    
    def __init__(self,input_sizes,mem_size,src_names=None,mem_name=None):
        if src_names is None: src_names = ["unnamed_input_%i"%i for i in xrange(len(input_sizes))]
        if mem_name is None: mem_name = "mem"

        Wiz_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wizs = [theano.shared(Wiz_val,name="Wiz_%s"%src_name) for (Wiz_val,src_name) in zip(Wiz_vals,src_names)]
        Wmz_val = normc(randnf(mem_size,mem_size))
        self.Wmz = theano.shared(Wmz_val,name="Wmz_%s"%mem_name)
        bz = np.zeros((1,mem_size),floatX)
        self.bz = theano.shared(bz,broadcastable=(True,False),name="bz_%s"%mem_name)

        Wir_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wirs = [theano.shared(Wir_val,name="Wir_%s"%src_name) for (Wir_val,src_name) in zip(Wir_vals,src_names)]
        Wmr_val = normc(randnf(mem_size,mem_size))
        self.Wmr = theano.shared(Wmr_val,name="Wmr_%s"%mem_name)
        br = np.zeros((1,mem_size),floatX)
        self.br = theano.shared(br,broadcastable=(True,False),name="br_%s"%mem_name)

        Wim_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wims = [theano.shared(Wim_val,name="Wim_%s"%src_name) for (Wim_val,src_name) in zip(Wim_vals,src_names)]
        Wmm_val = normc(np.eye(mem_size,dtype=floatX))
        self.Wmm = theano.shared(Wmm_val,name="Wmm_%s"%mem_name)
        bm = np.zeros((1,mem_size),floatX)
        self.bm = theano.shared(bm,broadcastable=(True,False),name="bm_%s"%mem_name)


    def __call__(self,M,*inputs):
        summands = [Xi.dot(Wiz) for (Xi,Wiz) in zip(inputs,self.Wizs)] + [M.dot(self.Wmz),self.bz]
        z = TT.nnet.sigmoid(TT.add(*summands))

        summands = [Xi.dot(Wir) for (Xi,Wir) in zip(inputs,self.Wirs)] + [M.dot(self.Wmr),self.br]
        r = TT.nnet.sigmoid(TT.add(*summands))

        summands = [Xi.dot(Wim) for (Xi,Wim) in zip(inputs,self.Wims)] + [(r*M).dot(self.Wmm),self.bm]
        Mtarg = TT.tanh(TT.add(*summands)) #pylint: disable=E1111

        Mnew = (1-z)*M + z*Mtarg
        return Mnew

    def opt_vars(self):
        out = []
        out.extend(self.Wizs)
        out.append(self.Wmz)
        out.append(self.bz)        
        out.extend(self.Wirs)
        out.append(self.Wmr)
        out.append(self.br)        
        out.extend(self.Wims)
        out.append(self.Wmm)
        out.append(self.bm)        
        return out


class RNN(ParameterizedFunc):
    """
    Wraps up some primitive "cell" and applies scan operator, so it gets multiple times to input.
    """
    def __init__(self, mem_size, input_size, cell_type="tanh", src_name = "input", mem_name="mem", truncate_gradient=False):
        if cell_type == "tanh":
            self.cell = TanhCell(input_sizes = [input_size], mem_size=mem_size, src_names=[src_name],mem_name=mem_name)
        elif cell_type=="gru":
            self.cell = GRUCell(input_sizes = [input_size], mem_size=mem_size, src_names=[src_name],mem_name=mem_name)
        else:
            raise NotImplementedError
        self.truncate_gradient=truncate_gradient
    def __call__(self,O,Minit,return_all=False):
        def onestep(o,m):
            nextm = self.cell(m,o)
            return nextm
        M_tnm,_ = theano.scan(fn=onestep, sequences=[O], outputs_info=[dict(initial=Minit,taps=[-1])],n_steps=O.shape[0],truncate_gradient=self.truncate_gradient or -1)
        return M_tnm if return_all else M_tnm[-1]


    def opt_vars(self):
        out = []
        out.extend(self.cell.opt_vars())
        return out



