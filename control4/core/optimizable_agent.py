from control4.misc.var_collection import VarCollection
from control4.misc.console_utils import Message
from control4.core.agent import Agent
import theano, theano.tensor as TT
import numpy as np

class OptimizableAgent(Agent):

    def __init__(self,_mdp):
        self.policy = VarCollection(self.policy_vars())
        self.vf = VarCollection(self.vf_vars())
        self.extras = VarCollection(self.extra_vars())

        self._input_names = self.input_info().keys()
        self._output_names = self.output_info().keys()

        for letter in "abqu" : assert letter in self._output_names

        input_dict = self.symbolic_inputs()
        input_list = self._input_list(input_dict)
        output_dict = self._symbolic_call(input_dict)
        output_list = [output_dict[name] for name in self._output_names]

        with Message("compiling agent func"):
            self._call = theano.function(input_list, output_list, on_unused_input='ignore')

    ################################
    # Abstract: inherited

    def lag_array_names(self):
        raise NotImplementedError

    def input_info(self):
        raise NotImplementedError

    def output_info(self):
        raise NotImplementedError

    def initialize_lag_arrays(self):
        raise NotImplementedError

    ################################
    # Abstract: new here

    def ponder(self, input_dict):
        """
        return a
        """
        raise NotImplementedError

    def cpd(self):
        """
        Conditional probability distribution of policy.
        """
        raise NotImplementedError

    def b2u(self,b):
        raise NotImplementedError

    def policy_vars(self):
        raise NotImplementedError

    def vf_vars(self):
        return []

    def extra_vars(self):
        return []

    ################################
    # Overrides

    def call(self, input_dict):
        outputs = self._call(*self._input_list(input_dict))
        return zip(self._output_names, outputs)

    ################################
    # New public methods

    def to_h5(self,grp):
        for vc in [self.policy, self.vf, self.extras]:
            vc.to_h5(grp)

    def from_h5(self,grp):
        for vc in [self.policy, self.vf, self.extras]:
            vc.from_h5(grp)

    def symbolic_inputs(self):
        return {name:TT.matrix(name=name,dtype=dtype) for (name,(_,dtype)) in self.input_info().iteritems()}

    def pprint(self):
        from tabulate import tabulate
        def bigeig(m):
            if m.ndim == 2:
                _,s,_ = np.linalg.svd(m)
                s.sort()
                return s[-1]
            else:
                return None

        def print_vars(title,varlist):
            print "*** %s parameters ***"%title
            print tabulate([(par.name, str(parval.shape), parval.mean(), parval.std(), bigeig(parval)) 
                for par in varlist for parval in (par.get_value(borrow=True),)],headers=["name","shape","mean","std","topsv"])
            print "*************************"
        print_vars("Agent",self.policy_vars())
        if len(self.vf_vars())>0: print_vars("Value Function",self.vf_vars())
    ################################
    # New private methods

    def _symbolic_call(self, input_dict, stochastic=True):
        """
        call with symbolic arrays
        """
        cpd = self.cpd()
        output = self.ponder(input_dict)
        if stochastic:
            b = cpd.draw(output["a"])
        else:
            b = cpd.mls(output["a"])
        output.update({
            "b":b,
            "q":cpd.liks(output["a"],b),
            "u":self.b2u(b)
        })
        return output

    def _input_list(self, input_dict):
        return [input_dict[name] for name in self._input_names]
