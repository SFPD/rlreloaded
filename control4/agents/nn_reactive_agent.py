from control4.core.optimizable_agent import OptimizableAgent
from control4.core.cpd import DiagonalGaussian
from control4.config import floatX, resolve_cfg_loc
from control4.nn.nn import MLP
from control4.algs.alg_params import string2dict
import json, numpy as np
import theano,theano.tensor as TT #pylint: disable=F0401


class NNReactiveAgent(OptimizableAgent):

    def __init__(self,mdp,policy_cfg, vf_cfg):

        self._has_vf = vf_cfg is not None

        pol_hid_sizes = policy_cfg.pop("hidden_layer_sizes",[25,25])
        pol_nonlinearity = policy_cfg.pop("nonlinearity","soft_rect")
        initial_stdev = policy_cfg.pop("initial_stdev", 0.5)
        self.clip_features = policy_cfg.pop("clip_features",False)
        self.clip_value = policy_cfg.pop("clip_value",5.0)
        self.vf_scaling = policy_cfg.pop("vf_scaling",100.0)
        if len(policy_cfg)>0:
            print "WARNING: didn't use parameters %s"%policy_cfg.keys()

        input_size = mdp.output_size("o")
        n_net_out = mdp.input_size("u")

        try:
            lo,hi = mdp.obs_ranges()
        except (NameError,NotImplementedError):
            print "NNReactiveAgent: generating scaling data"
            from control4.agents.gen_scaling_data import gen_scaling_data
            lo,hi = gen_scaling_data(mdp)
            lo -= 0.01 # TODO: Fix feature selection problem properly
            hi += 0.01


        input_trans = ((hi+lo)/2)[None,:]
        input_scaling = ((hi-lo)/2)[None,:]
        assert np.allclose( (np.array([lo,hi])- input_trans)/input_scaling,np.array([[-1],[1]]),atol=1e-3)
        self.input_scaling = theano.shared(input_scaling,name="input_scaling")
        self.input_trans = theano.shared(input_trans,name="input_trans")
        self.input_scaling.type.broadcastable=(True,False)
        self.input_trans.type.broadcastable=(True,False)

        lo,hi = mdp.ctrl_bounds()
        self.output_scaling = (hi-lo)/2
        self.output_trans = (hi+lo)/2
        assert np.allclose(self.output_scaling[None,:]*np.array([[-1],[1]])+self.output_trans[None,:],np.array([lo,hi]))

        self._cpd = DiagonalGaussian( n_net_out )


        self._input_info = {
            "o" : mdp.output_info()["o"]
        }

        self._output_info = {
            "a" : (n_net_out, floatX),
            "b" : (n_net_out,floatX),
            "q" : (None, floatX),
            "u" : (n_net_out, floatX)
        }
        if self._has_vf: self._output_info["v"] = (None,floatX)



        pol_n_hid = len(pol_hid_sizes)
        init_col_norms = [1.0] * pol_n_hid + [0.01]
        # Hidden layers
        self.net = MLP([input_size] + pol_hid_sizes + [n_net_out],[pol_nonlinearity]*pol_n_hid + ["none"],
            ["input"] + ["hid%i"%i for i in xrange(pol_n_hid)] + ["output"],init_col_norms=init_col_norms)

        if self._has_vf:
            vf_hid_sizes = vf_cfg.pop("hidden_layer_sizes",[25,25])
            vf_n_hid = len(vf_hid_sizes)
            vf_nonlinearity = vf_cfg.pop("nonlinearity","soft_rect")
            if len(vf_cfg)>0:
                print "WARNING: didn't use parameters %s"%vf_cfg.keys()        
            init_col_norms = [1.0] * vf_n_hid + [0.01]
            self.vf_net = MLP([input_size] + vf_hid_sizes + [1],[vf_nonlinearity]*vf_n_hid + ["none"],
                ["vfinput"] + ["vfhid%i"%i for i in xrange(vf_n_hid)] + ["vfoutput"],init_col_norms=init_col_norms)

        stdev_init = np.ones(n_net_out,floatX)*initial_stdev
        self.logstdev = theano.shared(np.log(stdev_init),name="logstdev")

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
        o = input_dict["o"]
        o = (o - self.input_trans) / self.input_scaling
        if self.clip_features:
            o = o.clip(-self.clip_value, self.clip_value)
        # std = TT.repeat(TT.exp(self.logstdev.reshape((1,-1))), o.shape[0], axis=0)
        std = TT.exp(self.logstdev.reshape((1,-1))) + TT.zeros([o.shape[0],1],floatX)
        out = {"a":TT.concatenate([self.net(o),std],axis=1)}
        if self._has_vf:
            out["v"] = self.vf_net(o)[:,0]*self.vf_scaling
        return out

    def cpd(self):
        return self._cpd

    def b2u(self,b_nb):
        # because b[0] is in [0,1,2], but action.horiz is in [-1,0,1]
        return b_nb*self.output_scaling + self.output_trans

    def policy_vars(self):
        return self.net.opt_vars() + [self.logstdev]

    def vf_vars(self):
        return self.vf_net.opt_vars() if self._has_vf else []

    def extra_vars(self):
        return self.net.extra_vars() + [self.input_scaling, self.input_trans]



def construct(params,mdp):
    if params.get("policy_cfg"):
        with open(resolve_cfg_loc(params["policy_cfg"]),"r") as fh:
            policy_cfg = json.load(fh)
    else:
        policy_cfg = {}
    policy_cfg.update(string2dict(params.get("policy_kws","")))

    if params.get("vf_opt_mode") == "separate":
        if params.get("vf_cfg"):
            with open(resolve_cfg_loc(params["vf_cfg"]),"r") as fh:
                vf_cfg = json.load(fh)
        else:
            vf_cfg = {}
        vf_cfg.update(string2dict(params.get("vf_kws","")))
    else:
        vf_cfg = None


    return NNReactiveAgent(mdp, policy_cfg, vf_cfg)
