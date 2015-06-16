#!/usr/bin/env python
import matplotlib.pyplot as plt
import theano #pylint: disable=F0401
import copy
import numpy as np
from control4.misc.console_utils import Message
from control4.algs.save_load_utils import load_agent_and_mdp,get_mdp,construct_agent
from control4.algs.advantage_est import demean_timeserieses
from control4.misc.randomness import random_indices
from control4.config import floatX #pylint: disable=W0611
from control4.core.rollout import rollout
from control4.maths import symbolic
import theano.tensor as TT

# ipython --pylab  -i gradlogpconvadv.py -- --num_trajs=1000  --max_steps=200 --horizon=50 --agent_module=control4.agents.nn_reactive_agent  --mdp_name=mjc:3swimmer

def pairwise_correlate(x_sm, y_tn,mode='valid'):
    """
    Use FFT to compute correlation between pairs of channels of x and y
    """
    S,M = x_sm.shape
    T,N = y_tn.shape
    U = S+T-1 # if mode==valid we can use less padding
    px_um = np.concatenate([x_sm,np.zeros((U-S,M))])
    py_un = np.concatenate([y_tn,np.zeros((U-T,N))])

    qpx_um = np.fft.fft(px_um,axis=0) #pylint: disable=E1103,E1101
    qpy_un = np.fft.fft(py_un,axis=0) #pylint: disable=E1103,E1101

    qconv_umn = qpx_um[:,:,None] * np.conj(qpy_un[:,None,:])
    conv_umn = np.fft.ifft(qconv_umn,axis=0).real #pylint: disable=E1103,E1101
    if mode == "valid":
        assert T<S
        return conv_umn[:S-T+1]
    else:
        raise NotImplementedError

def test_pairwise_correlate():
    x = np.random.randn(10,3)
    y = np.random.randn(8,2)

    corr0 = pairwise_correlate(x,y,'valid')
    corr1 = np.empty((x.shape[0] - y.shape[0] + 1, x.shape[1],y.shape[1]))
    for (i,xcol) in enumerate(x.T):
        for (j,ycol) in enumerate(y.T):
            corr1[:,i,j] = np.correlate(xcol,ycol,mode='valid')
    assert np.allclose(corr0,corr1,atol=1e-7)


def make_gradlogps(mdp,agent):
    o = TT.matrix("o",mdp.output_dtype("o"))
    b = TT.matrix("b",agent.output_dtype("b"))



    newa = agent.ponder({"o":o})["a"]
    logp_n = agent.cpd().logliks(newa, b)

    def onegrad(i):
        logp1 = theano.clone(logp_n, replace = {b:b[i:i+1],o:o[i:i+1]})[0]
        return symbolic.flatten(TT.grad(logp1, agent.policy_vars()))

    gradlogps,_ = theano.map(onegrad, TT.arange(logp_n.shape[0]))

    f = theano.function([o,b],gradlogps)
    return f

def main():
    # test_pairwise_correlate()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf")
    parser.add_argument("--agent_module")
    parser.add_argument("--mdp_name")
    parser.add_argument("--horizon",type=int,default=20)
    parser.add_argument("--num_trajs",type=int,default=100)
    parser.add_argument("--max_steps",type=int,default=500)
    parser.add_argument("--load_idx",type=int,default=-1)
    parser.add_argument("--plot_mode",choices=["off","save","interactive"],default="off")
    parser.add_argument("--plot_save_prefix")
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--outfile")
    # ValueParams.add_to_parser(parser)
    
    np.random.seed(0)
    args = parser.parse_args()

    assert bool(args.hdf) != (bool(args.agent_module) and bool(args.mdp_name))
    if args.hdf:    
        agent, mdp, _hdf = load_agent_and_mdp(args.hdf,args.load_idx)
    elif args.agent_module:
        mdp = get_mdp(args.mdp_name)
        agent = construct_agent({"agent_module":args.agent_module},mdp)
    
    fs_gradlogps = make_gradlogps(mdp,agent)
    n_params = agent.policy.size()

    horizon=args.horizon

    from collections import namedtuple
    Path = namedtuple("Path",['o','c','b'])


    save_arrs = ["o","b","c"]
    with Message("Doing rollouts"):
        paths = []
        for i_path in xrange(args.num_trajs):
            if i_path % 20 == 0:
                print "%i/%i done"%(i_path,args.num_trajs)
            init,traj = rollout(mdp, agent, args.max_steps,save_arrs=save_arrs)
            o = np.concatenate([init["o"]]+traj["o"][:-1])
            c = np.concatenate(traj["c"])
            b = np.concatenate(traj["b"])
            paths.append(Path(o,c,b))

    # vf = LinearVF(use_m=False, use_o=True, legendre_degree = 2, use_product_features=False)
    # vf.make_funcs()

    # with Message("Fitting value function"):        
    #     fit_linear_vf_single_path(vf, paths, args)


    li_c_t = [path.c.sum(axis=1) for path in paths]
    li_dmc_t = copy.deepcopy(li_c_t)
    demean_timeserieses(li_dmc_t)
    # li_delta_t = []
    # for (path,c_t) in zip(paths,li_c_t):
    #     v_t = vf.fs_vfunc(path.prevm_tg,path.o_tf)
    #     li_delta_t.append( c_t + args.gamma*v_t[1:] - v_t[:-1] )

    li_serieses = zip(li_dmc_t)
    series_names=["demeaned costs"]
    n_series = len(series_names)

    li_corr = [np.zeros((horizon,n_params)) for _ in xrange(n_series)]
    corr_tkz = np.zeros((horizon,n_series,n_params))
    sqcorr_tkz = np.zeros((horizon,n_series,n_params))

    count = 0
    for (i_path,path,serieses) in zip(xrange(len(paths)),paths,li_serieses):
        if i_path % 20 == 0:
            print "%i/%i done"%(i_path,len(paths))
        sig_tk = np.array(serieses).T
        grad_tz = fs_gradlogps(path.o,path.b)
        newcorr_tzk = pairwise_correlate( sig_tk, grad_tz[:-horizon+1], mode='valid')
        corr_tkz += newcorr_tzk
        sqcorr_tkz += newcorr_tzk**2


        # for (li_series_t,corr_tz) in zip(li_li_series,li_corr):
        #     for z in xrange(n_params):
        #         corr_tkz[:,z] += scipy.signal.correlate(li_series_t[i_path], grad_tz[:-horizon+1,z],mode='valid')

        # count += (grad_tz.shape[0]-horizon)
        count += 1
    
    corr_tkz /= count
    sqcorr_tkz /= count
    
    stderr_tkz = np.sqrt( (sqcorr_tkz - corr_tkz**2)/len(paths) )
    # NOTE stderr is not totally legit


    plot_stderr = True
    zs = random_indices(n_params,30)

    # plot_stderr = False
    # zs = np.arange(n_params)

    for (i_series,_corr_tz) in enumerate(li_corr):
        plt.figure(i_series+1)
        plt.clf()
        plt.title(series_names[i_series])
        for z in zs:
            line, = plt.plot(corr_tkz[:,i_series,z])
            if plot_stderr: plt.fill_between(np.arange(horizon), corr_tkz[:,i_series,z] - stderr_tkz[:,i_series,z],corr_tkz[:,i_series,z] + stderr_tkz[:,i_series,z],alpha=.1,color=line.get_color()) 

    if args.outfile: plt.savefig(args.outfile)

if __name__ == "__main__":
    main()