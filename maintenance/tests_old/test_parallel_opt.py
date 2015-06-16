import argparse
import numpy as np
from cloud.cloud_interface import create_cloud,load_cloud_config,get_slave_names
from cloud.cluster_pool import ClusterPool
from cloud.slave_loop import slave_loop
from control3.common_util import chunk_slices
from control3.parallel import sum_count_reducer
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

class G: #pylint: disable=W0232
    X = None
    y = None


######### Pure functions #######
L2COEFF = 1e-4
def fc(w, X):
    return 1. / (1. + np.exp(-X.dot(w))) 
def floss(w,X,z):
    N = X.shape[0]
    c = fc(w,X)
    EPSILON=1e-30
    return -(z*np.log(c+EPSILON) + (1-z)*np.log(1-c+EPSILON)).sum() + 0.5*L2COEFF*N*w.dot(w),N
def fgradloss(w,X,z):
    N = X.shape[0]
    c = fc(w,X)
    return ((c-z).reshape(-1,1)* X ).sum(axis=0) + L2COEFF*N*w,N
##################################


def f((w,sli)):    
    return floss(w,G.X[sli],G.y[sli])

def gradf((w,sli)):
    return fgradloss(w,G.X[sli],G.y[sli])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slave_addr",type=str)
    args = parser.parse_args()

    # The digits dataset
    digits = datasets.load_digits()
    G.X = digits['data']
    G.y = digits['target']


    if args.slave_addr:
        slave_loop(args.slave_addr)
    else:
        cloud_config = load_cloud_config()
        cloud = create_cloud(cloud_config)
        cluster = "testparopt"
        cloud.start_instances(instance_names=get_slave_names(3,instance_prefix=cluster))
        pool = ClusterPool(cloud,cluster,start_mode="the_prestige")
        slis = chunk_slices(G.X.shape[0], pool.size())

        w = np.zeros(G.X.shape[1])
        wslis = [(w,sli) for sli in slis]
        loss,losscount = pool.mapreduce(f,sum_count_reducer,wslis)
        grad,gradcount = pool.mapreduce(gradf,sum_count_reducer,wslis)

        loss1,losscount1 = f((w,slice(0,None,None))) 
        grad1,gradcount1 = gradf((w,slice(0,None,None))) 

        assert np.allclose(loss,loss1)
        assert np.allclose(losscount,losscount1)
        assert np.allclose(grad,grad1)
        assert np.allclose(gradcount,gradcount1)
