from control4.config import floatX
from control4.maths.numeric import uniform
from control4.core.agent import Agent

class RandomContinuousAgent(Agent):

    def __init__(self,mdp):
        assert mdp.input_dtype("u") == floatX
        self._u_info = mdp.input_info()["u"]
        self.bounds = mdp.ctrl_bounds()
    def lag_array_names(self):
        return []

    def input_info(self):
        return {}

    def output_info(self):
        return {}

    def call(self, input_dict):        
        lo,hi = self.bounds
        u = uniform(lo,hi).astype(floatX).reshape(1,-1)
        return {"u":u}

    def initialize_lag_arrays(self):
        return {}

def construct(_params,mdp):
    return RandomContinuousAgent(mdp)