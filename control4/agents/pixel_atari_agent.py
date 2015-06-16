from control4.core.optimizable_agent import OptimizableAgent
from control4.core.cpd import FactoredCategoricalDistribution
from control4.config import floatX, resolve_cfg_loc
from control4.algs.alg_params import string2dict
from control4.nn.nn import DenseLayer
from control4.nn.cnn import ConvLayer,SepChannelLayer
import json, numpy as np
import theano.tensor as TT #pylint: disable=F0401

def build_convnet(cfg,img_shape,out_size,prefix=""):
    cfg = cfg.copy()
    conv_layer_infos = cfg.pop("conv_layers")
    dense_layer_infos = cfg.pop("dense_layers")
    dense_layer_infos[-1]["output_size"] = out_size
    transition_mode = cfg.pop("transition_mode","dense")


    if len(cfg) > 0: print "WARNING: Didn't use parameters %s"%cfg.keys()        
    conv_layers = []
    prev_output_shape = img_shape
    prev_name = prefix+"obs"
    for (i,info) in enumerate(conv_layer_infos):
        targ_name = prefix+"conv%i"%i
        layer = ConvLayer(prev_output_shape,info["n_channels"],info["filter_size"],info["nonlinearity"],info["pool_shape"],info["subsample_shape"],src_name=prev_name,targ_name=targ_name)
        conv_layers.append(layer)
        prev_output_shape = layer.output_shape()
        prev_name = targ_name

    dense_layers = []
    if transition_mode == "dense":
        prev_output_size = np.prod(prev_output_shape)
    elif transition_mode == "com":
        prev_output_size = 2*prev_output_shape[0]
    elif transition_mode == "sep-channels":
        pass
    else:
        raise NotImplementedError

    for (i,info) in enumerate(dense_layer_infos):
        targ_name = prefix+"dense%i"%i
        output_size = info["output_size"]
        if i==0 and transition_mode == "sep-channels":
            layer = SepChannelLayer(prev_output_shape,output_size,info["nonlinearity"])
        else:
            layer = DenseLayer([prev_output_size],output_size, info["nonlinearity"],src_names=[prev_name],targ_name=targ_name)

        dense_layers.append(layer)
        prev_output_size = output_size
        prev_name = targ_name


    return conv_layers, dense_layers

def apply_convnet(input_dict, conv_layers, dense_layers, cfg):
    transition_mode = cfg.pop("transition_mode","dense") # XXX
    y = input_dict["o"].reshape([-1]+list(conv_layers[0].input_shape()))
    y = TT.cast(y,floatX)
    y = (y-128.0)/128.0
    for layer in conv_layers:
        y = layer(y)
    if transition_mode == "dense":
        y = y.flatten(2)
    elif transition_mode == "com":
        assert len(conv_layers) > 0
        _,height,width = layer.output_shape() #pylint: disable=W0631
        y = TT.exp(y) # ncyx
        y = y / y.sum(axis=2,keepdims=True).sum(axis=3,keepdims=True)
        cm0 = (y * (TT.arange(height,dtype=floatX)/TT.cast(height,floatX)-0.5)[None,None,:,None]).sum(axis=2,keepdims=True).sum(axis=3,keepdims=True)
        cm1 = (y * (TT.arange(width,dtype=floatX)/TT.cast(width,floatX)-0.5)[None,None,None,:]).sum(axis=2,keepdims=True).sum(axis=3,keepdims=True)
        y = TT.concatenate([cm0.flatten(2),cm1.flatten(2)],axis=1)
    elif transition_mode == "sep-channels":
        pass
    else:
        raise NotImplementedError 

    for layer in dense_layers:
        y = layer(y)

    return y

class PixelAtariAgent(OptimizableAgent):

    def __init__(self,mdp,policy_cfg,vf_cfg):

        assert mdp.obs_mode == "image"


        self.is_factored = policy_cfg.pop("is_factored",True)
        self.add_extra_factor = policy_cfg.pop("add_extra_factor",False)
        if self.is_factored:
            self._cpd = FactoredCategoricalDistribution( (3,3,2,4) if self.add_extra_factor else (3,3,2) )
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

        # Hidden layers

        img_shape = self.img_shape = mdp.img_shape()
        self.conv_layers, self.dense_layers = build_convnet(policy_cfg,img_shape, self._cpd.a_dim(),prefix='pol_')
        self.policy_cfg = policy_cfg

        ### XXX
        W = self.dense_layers[-1].Ws[0].get_value(borrow=True)
        W[:,:8] *= .01
        ###

        if vf_cfg is not None:
            self._has_vf = True
            self._output_info["v"] = (None,floatX)
            self.vf_conv_layers, self.vf_dense_layers = build_convnet(vf_cfg,img_shape, 1,prefix='vf_')
            self.vf_cfg = vf_cfg
        else:
            self._has_vf = False

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


        y = apply_convnet(input_dict, self.conv_layers, self.dense_layers, self.policy_cfg)
        ey = TT.exp(y) #pylint: disable=E1111
        def normalize(x):
            return x/x.sum(axis=1,keepdims=True)
        if self.is_factored:
            a = TT.concatenate([
                normalize(ey[:,start:end]) for (start,end) in self._cpd.slice_starts_ends],axis=1)
        else:
            a = normalize(ey)
        out = {"a": a}

        if self._has_vf: out["v"] = apply_convnet(input_dict, self.vf_conv_layers, self.vf_dense_layers, self.vf_cfg)[:,0]
        return out

    def cpd(self):
        return self._cpd

    def b2u(self,b_nb):
        # get rid of extra factor if necessary
        # then translate, because b[0] is in [0,1,2], but action.horiz is in [-1,0,1]
        return b_nb[:,:3] - np.array([[1,1,0]],'int64')

    def policy_vars(self):
        out = []
        for layer in self.conv_layers + self.dense_layers:
            out.extend(layer.opt_vars())
        return out

    def vf_vars(self):
        if self._has_vf:
            out = []
            for layer in self.vf_conv_layers+self.vf_dense_layers:
                out.extend(layer.opt_vars())
            return out
        else:
            return []

    def extra_vars(self):
        return []


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

    return PixelAtariAgent(mdp,policy_cfg,vf_cfg)
