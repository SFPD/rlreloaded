import numpy as np
from control4.maths.discount import discount
from control4.config import floatX

def demean_timeserieses(li_x_t):    
    baseline_t = rowwise_mean_of_cols(li_x_t)
    sorted_ts = sorted(len(x_t) for x_t in li_x_t)
    t_depletion = sorted_ts[max(0, len(sorted_ts)-5)]
    baseline_t[t_depletion:] = baseline_t[t_depletion-1]
    for x_t in li_x_t:
        x_t -= baseline_t[:len(x_t)]


def compute_advantages(paths,fs_vfunc,lam,gamma,demean=False,standardize=False):
    # Compute advantages
    # Simple method
    # XXX should properly use memory state
    li_v_t = [fs_vfunc(np.zeros((path.o_tf.shape[0],0),floatX),path.o_tf) for path in paths] # XXX doesn't use memory state
    for (path,v_t) in zip(paths,li_v_t):
        if path.done:
            v_t[-1]=0
    li_c_t = [path.c_tv[:-1].sum(axis=1) for path in paths]
    assert len(li_c_t)==len(li_v_t)
    li_delta_t = [c_t+gamma*v_t[1:] - v_t[:-1] for (c_t,v_t) in zip(li_c_t,li_v_t)]
    li_adv_t = [discount(delta_t, gamma*lam) for (delta_t) in li_delta_t]
    baseline_t = compute_baseline(li_adv_t)
    if demean: 
        for adv_t in li_adv_t:
            adv_t -= baseline_t[:len(adv_t)]

    if standardize:
        std = np.concatenate(li_adv_t).std()
        for adv_t in li_adv_t:
            adv_t /= std

    return li_adv_t

def compute_baseline(li_x_t):
    baseline_t = rowwise_mean_of_cols(li_x_t)
    sorted_ts = sorted(len(x_t) for x_t in li_x_t)
    t_depletion = sorted_ts[max(0, len(sorted_ts)-5)]
    baseline_t[t_depletion:] = baseline_t[t_depletion-1]
    return baseline_t    

def rowwise_mean_of_cols(cols):
    """
    Given a list of columns, possibly of different lengths, compute mean of each row
    """
    maxlen = max(len(col) for col in cols)
    sums = np.zeros(maxlen,floatX)
    counts = np.zeros(maxlen,'int32')
    for col in cols:
        sums[:len(col)] += col
        counts[:len(col)] += 1
    return sums/counts
