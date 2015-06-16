import numpy as np
from control4.core.agent import Agent

class RandomDiscreteAgent(Agent):

    def __init__(self,num_actions):
        self.num_actions = num_actions

    def lag_array_names(self):
        return []

    def input_info(self):
        return {}

    def output_info(self):
        return {}

    def call(self, input_dict):     
        return {"u":np.random.randint(low=0,high=self.num_actions,size=(1,1))}

    def initialize_lag_arrays(self):
        return {}

def construct(_params,mdp):
    return RandomDiscreteAgent(mdp.num_actions())