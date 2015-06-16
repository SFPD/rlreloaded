from control4.core.mdp import MDP
import theano, theano.tensor as TT

class SymbolicMDP(MDP):
    
    """
    MDP for which call() is defined by symbolic operation (in theano)
    """

    def __init__(self):
        x = TT.matrix(name="x",dtype=self.input_dtype("x"))
        u = TT.matrix(name="u",dtype=self.input_dtype("u"))
        outputs = self.symbolic_call(x,u)
        self._call = theano.function([x,u],outputs)

    ################################


    # New

    def symbolic_call(self,x,u):
        raise NotImplementedError

    def output_names(self):
        raise NotImplementedError

    
    # Inherited

    def initialize_mdp_arrays(self):
        raise NotImplementedError

    def input_info(self):
        raise NotImplementedError

    def output_info(self):
        raise NotImplementedError

    def plot(self, mdp_arrs, pol_arrs):
        raise NotImplementedError

    def cost_names(self):
        raise NotImplementedError

    ################################

    def call(self,input_arrs):
        outputs = self._call(input_arrs["x"],input_arrs["u"])
        return dict(zip(self.output_names(),outputs))




