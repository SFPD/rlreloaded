import numpy as np
import scipy.signal

def discount(x, gamma):
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def discountedsum(x, gamma, axis=0):
    """
    x[0] + gamma*x[1] + ....
    """
    x = np.rollaxis(x, axis)
    return discount(x, gamma)[0]


def test_discount():
    arr = np.random.randn(10)
    assert np.allclose(np.correlate(arr,0.9**np.arange(10),mode='full')[9:],discount(arr,0.9))    

    arr = np.random.randn(3,4,5)
    assert np.allclose(discountedsum(arr,.9,axis=0) , discount(arr,.9)[0])
    assert np.allclose(discountedsum(arr,.9,axis=1) , discount(arr.transpose(1,0,2),.9)[0])
    assert np.allclose(discountedsum(arr,.9,axis=2) , discount(arr.transpose(2,0,1),.9)[0])

if __name__ == "__main__":
    test_discount()