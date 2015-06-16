from control4.core.optimizable_agent import OptimizableAgent
from control4.maths.symbolic import rbf
from control4.core.cpd import FactoredCategoricalDistribution
from control4.config import floatX, resolve_cfg_loc
import json, numpy as np,itertools
import theano,theano.tensor as TT

def grid_points(bounds, res, wraps=None):
    if wraps is None: wraps = (False,)*len(bounds)
    return np.array(list(itertools.product(*[np.linspace(lo,hi,res,endpoint = ~w) for ((lo,hi),w) in zip(bounds.T,wraps) ])),floatX)

class RBFDiscreteAgent(OptimizableAgent):

    def __init__(self,policy_cfg,mdp):

        input_dim = mdp.output_size("o")
        output_dim = mdp.input_size("u")

        self._normalize = policy_cfg.pop("normalize",False)        
        self._input_reses = policy_cfg.pop("input_reses",(25,)*input_dim)
        self._output_reses = policy_cfg.pop("output_reses",(3,)*output_dim)
        if len(policy_cfg)>0:
            print "WARNING: didn't use parameters %s"%policy_cfg.keys()
        self._wraps = mdp.obs_wraps() if hasattr(mdp,"obs_wraps") else [False]*output_dim
        self._input_bounds = mdp.obs_bounds()
        self._output_bounds = mdp.ctrl_bounds()

        u_dim = mdp.input_size("u")

        self._cpd = FactoredCategoricalDistribution(self._output_reses )
        self._input_info = {
            "o" : mdp.output_info()["o"]
        }

        n_net_out = np.sum(self._output_reses)

        self._output_info = {
            "a" : (n_net_out, floatX),
            "b" : (n_net_out,'int64'),
            "q" : (None, floatX),
            "u" : (u_dim, floatX)
        }

        self.w_fa = theano.shared(np.zeros((np.prod(self._input_reses),n_net_out),floatX),name='w')
        self.b_a = theano.shared(np.zeros((n_net_out,),floatX),name='b')

        OptimizableAgent.__init__(self,mdp)

    def lag_array_names(self):
        return []

    def input_info(self):
        return self._input_info

    def output_info(self):
        return self._output_info

    def initialize_lag_arrays(self):
        return {}

    ################################

    def ponder(self, input_dict):
        o = input_dict.get("o")
        f_nf = rbf(o, self._input_bounds, self._input_reses, self._wraps, self._normalize)
        y_ny = f_nf.dot(self.w_fa) + self.b_a[None,:]
        ey_ny = TT.exp(y_ny) #pylint: disable=E1111
        output_res = self._output_reses[-1]
        output_dim = len(self._output_reses)
        ey_ner = ey_ny.reshape((ey_ny.shape[0], output_dim, output_res))
        ney_ner = ey_ner/ey_ner.sum(axis=2,keepdims=True)
        nea_nk = ney_ner.reshape((ney_ner.shape[0], output_res*output_dim))
        return {"a":nea_nk}

    def cpd(self):
        return self._cpd

    def b2u(self,b_nb):
        lo_e,hi_e = self._output_bounds
        scalings_e = ((hi_e-lo_e)/(np.array(self._output_reses)-1)).astype(floatX)
        return TT.cast(b_nb,floatX) * scalings_e.reshape(1,-1) + lo_e.reshape(1,-1)

    def policy_vars(self):
        return [self.w_fa, self.b_a]



def construct(params,mdp):
    if params.get("policy_cfg"):
        with open(resolve_cfg_loc(params["policy_cfg"]),"r") as fh:
            policy_cfg = json.load(fh)
    else:
        policy_cfg = {}

    return RBFDiscreteAgent(policy_cfg,mdp)
