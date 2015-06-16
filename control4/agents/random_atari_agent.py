import numpy as np
from control4.core.agent import Agent

class RandomAtariAgent(Agent):

    def __init__(self):
        pass

    def lag_array_names(self):
        return []

    def input_info(self):
        return {}

    def output_info(self):
        return {}

    def call(self, input_dict):        
        horiz=np.random.randint(-1,2)
        vert=np.random.randint(-1,2)
        button=np.random.randint(0,2)

        return {"u":np.array([[horiz,vert,button]])}

    def initialize_lag_arrays(self):
        return {}

def construct(_params,_mdp):
    return RandomAtariAgent()