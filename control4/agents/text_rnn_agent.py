"""
Exactly copied from TextRNNAgent except for a few lines
"""

from control4.core.optimizable_agent import OptimizableAgent
from control4.core.cpd import FactoredCategoricalDistribution
from control4.config import floatX, resolve_cfg_loc
from control4.nn.nn import DenseLayer,LayerChain
from control4.nn.rnn import GRUCell
import json, numpy as np
import theano


class TextRNNAgent(OptimizableAgent):

    def __init__(self,policy_cfg,mdp):

        self.n_mem = n_mem = policy_cfg.pop("mem_size",20)
        nonlinearity = policy_cfg.pop("nonlinearity","soft_rect")
        hidden_layer_sizes = policy_cfg.pop("hidden_layer_sizes",[12])
        if len(policy_cfg)>0:
            print "WARNING: didn't use parameters %s"%policy_cfg.keys()


        input_size = mdp.output_size("o")
        n_net_out = mdp.num_actions()

        self.output_scaling = np.ones(n_net_out,floatX)
        self.output_trans = np.zeros(n_net_out,floatX)        

        self._cpd = FactoredCategoricalDistribution( [mdp.num_actions()] )


        self._input_info = {
            "o" : mdp.output_info()["o"],
            "m" : (n_mem, floatX)
        }

        self._output_info = {
            "a" : (n_net_out, floatX),
            "b" : (1,"int64"),
            "q" : (None, floatX),
            "u" : (n_net_out, "int64"),
            "m" : (n_mem, floatX)
        }

        self.mo2m = GRUCell([input_size],n_mem)

        layers = []
        # Hidden layers        
        layer_idx = 0
        prev_output_size = n_mem
        for output_size in hidden_layer_sizes:
            layers.append(DenseLayer([prev_output_size],output_size,nonlinearity,src_names=[str(layer_idx)],targ_name=str(layer_idx+1)))
            prev_output_size = output_size
            layer_idx += 1

        # Final layer
        output_size = n_net_out
        layers.append(DenseLayer([prev_output_size],output_size,"softmax",src_names=[str(layer_idx)],targ_name="output",col_norm=0.0))

        b = layers[-1].b.get_value(borrow=True)
        b.flat[-2:] -= 1.0
        self.net = LayerChain(layers)

        OptimizableAgent.__init__(self,mdp)

    def lag_array_names(self):
        return ["m"]

    def input_info(self):
        return self._input_info

    def output_info(self):
        return self._output_info

    def initialize_lag_arrays(self):
        return {"m":np.zeros((1,self.n_mem),floatX)}

    ################################

    def ponder(self, input_dict):
        o = input_dict["o"]
        m = input_dict["m"]
        newm = self.mo2m(m,o)
        a = self.net(newm)
        return {"a":a,"m":newm}

    def cpd(self):
        return self._cpd

    def b2u(self,b_nb):
        return b_nb

    def policy_vars(self):
        return self.mo2m.opt_vars() + self.net.opt_vars()

    def other_vars(self):
        return self.mo2m.other_vars() + self.net.other_vars()



def construct(params,mdp):
    if params.get("policy_cfg"):
        with open(resolve_cfg_loc(params["policy_cfg"]),"r") as fh:
            policy_cfg = json.load(fh)
    else:
        policy_cfg = {}

    return TextRNNAgent(policy_cfg,mdp)
