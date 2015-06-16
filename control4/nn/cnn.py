from control4.maths.symbolic import apply_nonlinearity
from control4.misc.console_utils import colorize
from control4.config import floatX
import theano,theano.tensor as TT #pylint: disable=F0401
from theano.tensor.nnet import conv #pylint: disable=F0401
import numpy as np
from control4.nn.nn import ParameterizedFunc,randn_init

def pool(x_ncuv, rows_in, cols_in, poolshp, pool_type='max'):
    if rows_in % poolshp[0] != 0 or cols_in % poolshp[0] != 0:
        row_residue = rows_in%poolshp[0]
        col_residue = cols_in%poolshp[1]
        print colorize("warning: image shape not divisible by pool size. cropping %i/%i on top, %i/%i on left"%(row_residue,rows_in,col_residue,cols_in) , 'yellow')
        x_ncuv = x_ncuv[:,:,:rows_in - row_residue, :cols_in - col_residue]
    x_ncpaqb = x_ncuv.reshape( (x_ncuv.shape[0], x_ncuv.shape[1], rows_in // poolshp[0], poolshp[0], cols_in // poolshp[1], poolshp[1]) )
    x_ncpqab = x_ncpaqb.dimshuffle(0,1,2,4,3,5)
    x_ncpq_ab = x_ncpqab.flatten(5)
    if pool_type == 'max':
        x_ncpq = x_ncpq_ab.max(axis=4)
    elif pool_type == 'mean':
        x_ncpq = x_ncpq_ab.mean(axis=4)
    elif pool_type == '2norm':
        x_ncpq = TT.sqrt(TT.square(x_ncpq_ab).sum(axis=4)) #pylint: disable=E1111
    elif pool_type == 'softmax':
        x_ncpq = TT.log(TT.exp(x_ncpq_ab).sum(axis=4)) #pylint: disable=E1111
    return x_ncpq

class SepChannelLayer(ParameterizedFunc):
    def __init__(self,input_shape,output_size,src_name=None,targ_name=None):
        if src_name is None: src_name = "sep_src"
        if targ_name is None: targ_name = "sep_targ"
        in_chans, in_rows, in_cols = input_shape
        self.in_chans = in_chans
        assert output_size % in_chans == 0
        W_init = np.array([randn_init(in_rows*in_cols,output_size/in_chans) for _ in xrange(in_chans)])
        b_init = np.zeros((1,output_size),floatX)
        self.W = theano.shared(W_init,name="W_%s_%s"%(src_name,targ_name))
        self.b = theano.shared(b_init,name="b_%s"%targ_name)
        self.b.type.broadcastable = (True,False)

    def __call__(self, img_ncxy):
        img_ncf = img_ncxy.flatten(3)
        return TT.stack(*[img_ncf[:,i,:].dot(self.W[i]) for i in xrange(self.in_chans)]).transpose([1,0,2]).flatten(2) + self.b

    def opt_vars(self):
        return [self.W, self.b]

class ConvLayer(ParameterizedFunc):

    def __init__(self,single_image_shape, n_channels, filter_size, nonlinearity, pool_shape, subsample_shape,src_name=None,targ_name=None):
        if src_name is None: src_name = "unnamedinput"
        if targ_name is None: targ_name = "unnamedoutput"
        self.image_shape = (None,)+single_image_shape
        filter_shape = self.filter_shape = [n_channels,single_image_shape[0]] + filter_size
        self.pool_shape=pool_shape
        self.subsample_shape=subsample_shape


        assert self.image_shape[1] == self.filter_shape[1]
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_shape))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            np.random.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=floatX),borrow=True,name="W_%s_%s"%(src_name,targ_name))

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=floatX)
        if nonlinearity == "hard_rect": b_values += .25        
        self.b = theano.shared(value=b_values, borrow=True,name='b_%s'%targ_name)

        self.nonlinearity = nonlinearity
        self.pool_type = "max"

    def __call__(self, xin):
        print "input:",xin.shape, "filters",self.filter_shape, "image:",self.image_shape
        conv_out = conv.conv2d(input=xin, filters=self.W,
                filter_shape=self.filter_shape, image_shape=self.image_shape,subsample=self.subsample_shape)
        
        if np.prod(self.pool_shape) > 1:
            co_rows, co_cols = self._conv_output_hw()
            pooled_out = pool(conv_out, co_rows, co_cols, self.pool_shape, pool_type = self.pool_type)
        else:
            print "skipping pooling"
            pooled_out = conv_out
        output = apply_nonlinearity(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'), self.nonlinearity) #pylint: disable=E1111

        return output

    def _conv_output_hw(self):
        in_hw = np.array(self.image_shape[-2:])
        return ceildiv(in_hw - np.array(self.filter_shape[-2:])+1 , self.subsample_shape)

    def input_shape(self):
        return self.image_shape[-3:]

    def output_shape(self):
        d0,d1 = np.floor_divide( self._conv_output_hw(), self.pool_shape)
        return (self.filter_shape[0], d0, d1)

    def opt_vars(self):
        return [self.W, self.b]


def ceildiv(x,y):
    x = np.array(x,floatX)
    y = np.array(y,floatX)
    z = np.ceil(x/y)
    return np.array(z,dtype=int)
