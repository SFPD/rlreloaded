from control4.core.optimizable_agent import OptimizableAgent
from control4.core.cpd import DiagonalGaussian
from control4.config import floatX, resolve_cfg_loc
from control4.nn.nn import DenseLayer,LayerChain
from control4.nn.rnn import GRUCell
import json, numpy as np
import theano,theano.tensor as TT


class NavRNNAgent(OptimizableAgent):

    def __init__(self,policy_cfg,mdp):

        self.n_mem = n_mem = policy_cfg.pop("mem_size",10)
        nonlinearity = policy_cfg.pop("nonlinearity","soft_rect")
        hidden_layer_sizes = policy_cfg.pop("hidden_layer_sizes",[10])
        if len(policy_cfg)>0:
            print "WARNING: didn't use parameters %s"%policy_cfg.keys()


        input_size = mdp.output_size("o")
        n_net_out = mdp.input_size("u")

        self.output_scaling = np.ones(n_net_out,floatX)
        self.output_trans = np.zeros(n_net_out,floatX)        

        self._cpd = DiagonalGaussian( n_net_out )


        self._input_info = {
            "o" : mdp.output_info()["o"],
            "m" : (n_mem, floatX)
        }

        self._output_info = {
            "a" : (n_net_out, floatX),
            "b" : (n_net_out,floatX),
            "q" : (None, floatX),
            "u" : (n_net_out, floatX),
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
        layers.append(DenseLayer([prev_output_size],output_size,"none",src_names=[str(layer_idx)],targ_name="output",col_norm=0))

        self.net = LayerChain(layers)

        stdev_init = np.ones(n_net_out,floatX)*.8
        self.logstdev = theano.shared(np.log(stdev_init),name="logstdev")

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
        # std = TT.repeat(TT.exp(self.logstdev.reshape((1,-1))), o.shape[0], axis=0)
        std = TT.exp(self.logstdev.reshape((1,-1))) + TT.zeros([o.shape[0],1],floatX)
        newm = self.mo2m(m,o)
        # y = self.net(TT.concatenate([newm,o],axis=1))
        y = self.net(newm)
        return {"a":TT.concatenate([y,std],axis=1),"m":newm}

    def cpd(self):
        return self._cpd

    def b2u(self,b_nb):
        # because b[0] is in [0,1,2], but action.horiz is in [-1,0,1]
        return b_nb*self.output_scaling + self.output_trans

    def policy_vars(self):
        return self.mo2m.opt_vars() + self.net.opt_vars() + [self.logstdev]

    def other_vars(self):
        return self.mo2m.other_vars() + self.net.other_vars()



def construct(params,mdp):
    if params.get("policy_cfg"):
        with open(resolve_cfg_loc(params["policy_cfg"]),"r") as fh:
            policy_cfg = json.load(fh)
    else:
        policy_cfg = {}

    return NavRNNAgent(policy_cfg,mdp)
