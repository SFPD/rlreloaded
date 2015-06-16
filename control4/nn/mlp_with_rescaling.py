from control4.nn.nn import DenseLayer,ElemwiseLinearLayer,ParameterizedFunc
import theano,theano.tensor as TT #pylint: disable=F0401
from control4.misc.console_utils import Message

class MLPWithRescaling(ParameterizedFunc):
    """
    A sequence of DenseLayer
    """
    def __init__(self, sizes, nonlinearities, names=None, init_col_norms=None):
        """
        sizes: number of units at each layer
        i.e., we have (sizes-1) weight matrices and nonlinearities
        """
        assert len(nonlinearities) == len(sizes)-1
        if names is None: names = [str(i) for i in xrange(len(sizes))]
        else: assert len(names) == len(sizes)
        if init_col_norms is None: init_col_norms = [1.0 for _ in xrange(len(nonlinearities))]
        else: assert len(init_col_norms) == len(nonlinearities)
        self.layers = []
        self.slayers = []
        prev_output_size = sizes[0]
        prev_name = names[0]
        for (output_size,nonlinearity,name,col_norm) in zip(sizes[1:],nonlinearities,names[1:],init_col_norms):
            slayer = ElemwiseLinearLayer(prev_output_size)
            self.slayers.append(slayer)
            layer = DenseLayer([prev_output_size],output_size,nonlinearity=nonlinearity,src_names=[prev_name],targ_name=name,col_norm=col_norm)
            self.layers.append(layer)

            prev_output_size=output_size
            prev_name = name

        X = TT.matrix("X")
        with Message("compiling activation stats func"):
            means,stds = self.activation_stats(X)
            self.fstats = theano.function([X],means+stds)

    def __call__(self,X):
        for (slayer,layer) in zip(self.slayers,self.layers):
            X = slayer(X)
            X = layer(X)
        return X

    def renormalize(self,X):
        means_stds = self.fstats(X)
        means = means_stds[:len(means_stds)//2]
        stds = means_stds[len(means_stds)//2:]
        for (slayer,layer,mean,std) in zip(self.slayers,self.layers,means,stds):
            slayer.update_with_compensation(-mean,1.0/(std+1e-4),layer.Ws[0],layer.b)

    def activation_stats(self,X):
        means = []
        stds = []
        for (slayer,layer) in zip(self.slayers,self.layers)[:1]:
            means.append(X.mean(axis=0))
            stds.append(X.std(axis=0))
            X = slayer(X)
            X = layer(X)
        return means,stds


    def opt_vars(self):
        out = []
        for layer in self.layers:
            out.extend(layer.opt_vars())
        return out

    def other_vars(self):
        out = []
        for (_,slayer) in zip(self.layers,self.slayers):
            out.extend(slayer.other_vars())
        return out
