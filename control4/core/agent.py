class Agent(object):

    def lag_array_names(self):
        """
        List of names of lagged outputs of the policy
        """
        raise NotImplementedError

    def input_info(self):
        """
        mapping from name -> (size, dtype)
        """
        raise NotImplementedError

    def output_info(self):
        """
        mapping from name -> (size, dtype)
        """
        raise NotImplementedError

    def call(self, input_dict):
        """
        call the policy and return a dict of results
        """
        raise NotImplementedError

    def initialize_lag_arrays(self):        
        raise NotImplementedError

    ################################

    def input_size(self,name):
        return self.input_info()[name][0]

    def input_dtype(self,name):
        return self.input_info()[name][1]        

    def output_size(self,name):
        return self.output_info()[name][0]        

    def output_dtype(self,name):
        return self.output_info()[name][1]


