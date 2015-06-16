from control4.core.optimizable_agent import OptimizableAgent
from control4.core.cpd import FactoredCategoricalDistribution
from control4.config import floatX, resolve_cfg_loc
from control4.algs.alg_params import string2dict
from control4.nn.nn import MLP
import json, numpy as np
import theano.tensor as TT #pylint: disable=F0401


class RamAtariAgent(OptimizableAgent):

    def __init__(self,mdp,policy_cfg,vf_cfg):

        self._has_vf = vf_cfg is not None

        pol_hid_sizes = policy_cfg.pop("hidden_layer_sizes",[20,20])
        pol_nonlinearity = policy_cfg.pop("nonlinearity","tanh")        
        self.is_factored = policy_cfg.pop("is_factored",True)
        if len(policy_cfg)>0:
            print "WARNING: didn't use parameters %s"%policy_cfg.keys()



        input_size = mdp.output_size("o")

        if self.is_factored:
            self._cpd = FactoredCategoricalDistribution( (3,3,2) )
        else:
            self._cpd = FactoredCategoricalDistribution( (5,) )

        pol_out_size = self._cpd.a_dim()

        self._input_info = {
            "o" : mdp.output_info()["o"]
        }

        self._output_info = {
            "a" : (pol_out_size, floatX),
            "b" : (self._cpd.b_dim(),'int64'),
            "q" : (None, floatX),
            "u" : (3, 'int64')
        }

        pol_n_hid = len(pol_hid_sizes)
        init_col_norms = [1.0] * pol_n_hid + [0.01]
        # Hidden layers
        self.net = MLP([input_size] + pol_hid_sizes + [pol_out_size],[pol_nonlinearity]*pol_n_hid + ["none"],
            ["input"] + ["hid%i"%i for i in xrange(pol_n_hid)] + ["output"],init_col_norms=init_col_norms)


        if vf_cfg is not None:
            vf_hid_sizes = vf_cfg.pop("hidden_layer_sizes",[20,20])
            vf_nonlinearity = vf_cfg.pop("nonlinearity","tanh")
            if len(vf_cfg)>0:
                print "WARNING: didn't use parameters %s"%vf_cfg.keys()

            self._output_info["v"] = (None,floatX)
            vf_n_hid = len(vf_hid_sizes)
            self.vf_net = MLP([input_size] + vf_hid_sizes + [1],[vf_nonlinearity]*vf_n_hid+ ["none"],
                ["vfinput"] + ["vfhid%i"%i for i in xrange(vf_n_hid)] + ["vfoutput"],
                init_col_norms=[1.0]*vf_n_hid + [0.01])
        else:
            self.vf_net = None

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
        o = TT.cast(o,floatX)
        o = (o-128.0)/128.0
        y = self.net(o)        
        ey = TT.exp(y) #pylint: disable=E1111

        def normalize(x):
            return x/x.sum(axis=1,keepdims=True)
        if self.is_factored:
            a = TT.concatenate([
                normalize(ey[:,0:3]),
                normalize(ey[:,3:6]),
                normalize(ey[:,6:8])],axis=1)
        else:
            a = normalize(ey)
        out = {"a": a}
        if self._has_vf: out["v"] = self.vf_net(o)[:,0]
        return out

    def cpd(self):
        return self._cpd

    def b2u(self,b_nb):
        # because b[0] is in [0,1,2], but action.horiz is in [-1,0,1]
        return b_nb - np.array([[1,1,0]],'int64')

    def policy_vars(self):
        return self.net.opt_vars()

    def vf_vars(self):
        return self.vf_net.opt_vars() if self._has_vf else []

    def extra_vars(self):
        return self.net.extra_vars()



def construct(params,mdp):
    if params.get("policy_cfg"):
        with open(resolve_cfg_loc(params["policy_cfg"]),"r") as fh:
            policy_cfg = json.load(fh)
    else: 
        policy_cfg = {}
    policy_cfg.update(string2dict(params.get("policy_kws","")))

    if params["vf_opt_mode"] == "separate":
        if params.get("vf_cfg"):
            with open(resolve_cfg_loc(params["vf_cfg"]),"r") as fh:
                vf_cfg = json.load(fh)
        else: 
            vf_cfg = {}
        vf_cfg.update(string2dict(params.get("vf_kws","")))
    else:
        vf_cfg = None

    return RamAtariAgent(mdp,policy_cfg,vf_cfg)
