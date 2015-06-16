from control4.core.agent import Agent
from control4.misc.console_utils import Message
import theano #pylint: disable=F0401

class MLAgent(Agent):
    """
    Wraps an OptimizableAgent and modifies it to take the most likely action
    """
    def __init__(self, oa):
        self.oa = oa

        input_dict = self.oa.symbolic_inputs()
        input_list = self.oa._input_list(input_dict)
        output_dict = self.oa._symbolic_call(input_dict,stochastic=False)
        output_list = [output_dict[name] for name in self.oa._output_names]

        with Message("compiling agent func"):
            self._call = theano.function(input_list, output_list, on_unused_input='ignore')


    def call(self,input_dict):
        outputs = self._call(*self.oa._input_list(input_dict))
        return zip(self.oa._output_names, outputs)

    def initialize_lag_arrays(self):
        return self.oa.initialize_lag_arrays()

    def output_info(self):
        return self.oa.output_info()

    def input_info(self):
        return self.oa.input_info()

    def lag_array_names(self):
        return self.oa.lag_array_names()