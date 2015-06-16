#!/usr/bin/env python
import matplotlib.pyplot as plt
from control4.core.rollout import rollout
import argparse
from control4.core.vines import sample_rollout_set,generate_branches
from control4.algs.save_load_utils import load_agent_and_mdp,get_mdp,construct_agent
from control4.misc.console_utils import Message
import numpy as np
from control4.maths.discount import discountedsum
import theano,theano.tensor as TT #pylint: disable=F0401

# ipython --pylab -i measure_response_fn.py -- --agent_module=control4.agents.nn_reactive_agent  --mdp_name=mjc:3swimmer --num_vines=10 --vine_length=400  --branch_length=200 --action_samples=2 --trials_per_action=1000 --rollout_set_size=10
def plot_mean_std(x_ts,plot_all,**plot_kws):
    idx_n = np.arange(x_ts.shape[0])
    s = x_ts.shape[1]
    xmean_t = x_ts.mean(axis=1)
    xstd_t = x_ts.std(axis=1)
    ax = plt.gca()
    ax.plot(xmean_t,lw=3,**plot_kws)
    if plot_all: ax.plot(x_ts,alpha=0.5,**plot_kws)
    # ax.fill_between(idx_n, xmean_t-xstd_t, xmean_t + xstd_t,alpha=0.25,facecolor=plot_kws.get('color','r'))
    ax.fill_between(idx_n, xmean_t-xstd_t/np.sqrt(s), xmean_t + xstd_t/np.sqrt(s),alpha=0.5,facecolor=plot_kws.get('color','r'))

def main():
# if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf")
    parser.add_argument("--agent_module")
    parser.add_argument("--mdp_name")
    parser.add_argument("--all_actions",action="store_true")
    parser.add_argument("--rollout_set_size",type=int,default=5)
    parser.add_argument("--vine_length",type=int,default=100)
    parser.add_argument("--num_vines",type=int,default=10)
    parser.add_argument("--branch_length",type=int,default=100)
    parser.add_argument("--action_samples",type=int,default=8)
    parser.add_argument("--trials_per_action",type=int,default=100)
    parser.add_argument("--load_idx",type=int,default=-1)
    parser.add_argument("--plot_all",action="store_true")
    parser.add_argument("--save_prefix")

    args = parser.parse_args()
    np.random.seed(0)

    assert bool(args.hdf) != (bool(args.agent_module) and bool(args.mdp_name))
    if args.hdf:    
        agent, mdp, _hdf = load_agent_and_mdp(args.hdf,args.load_idx)
    elif args.agent_module:
        mdp = get_mdp(args.mdp_name)
        agent = construct_agent({"agent_module":args.agent_module},mdp)
    
    b = TT.matrix("b",dtype=agent.output_dtype("b"))
    fs_b2u = theano.function([b],agent.b2u(b))

    with Message("making vines"):
        inittrajs = [rollout(mdp,agent,args.vine_length,save_arrs=("x","o","a")) for _ in xrange(args.num_vines)]
    rsd = sample_rollout_set(inittrajs,args.rollout_set_size)
    with Message("making branches"):
        bd = generate_branches(mdp, agent, rsd,
            branch_length = args.branch_length, 
            actions_per_state = args.action_samples,
            trials_per_action = args.trials_per_action, 
            all_actions = args.all_actions,
            fs_b2u = fs_b2u
        )

    # do_rollout_with_state(x, mdp, policy, onesl.f_policy, max_steps, callback=callback)
    c_tnks = np.concatenate([np.repeat(bd.cxy_nk[:,:,None,None],args.trials_per_action, axis=2),bd.cyz_nkst],axis=3).transpose(3,0,1,2)

    vf_enabled = "v" in agent.output_info()

    _,_,A,S = c_tnks.shape
    if vf_enabled:
        v_tnks = np.concatenate([np.tile(bd.vx_n[:,None,None,None],(1,A,S,1)),bd.vyz_nkst],axis=3).transpose(3,0,1,2)
        delta_tnas = c_tnks + v_tnks[1:] - v_tnks[:-1]

    for n in xrange(args.rollout_set_size):
        plt.clf()
        c_tks = c_tnks[:,n]
        c0_ts = c_tks[:,0]
        c1_ts = c_tks[:,-1]

        if vf_enabled:
            delta_tas = delta_tnas[:,n]
            delta0_ts = delta_tas[:,0]
            delta1_ts = delta_tas[:,-1]


        # plt.plot(c_tks[:,0,:],color='b')
        # mdp.plot(rsd.x_nd[n],np.zeros(mdp.ctrl_dim(),mdp.ctrl_dtype()))
        plt.subplot(2,1,1)
        plot_mean_std(c0_ts,args.plot_all,color='r')
        plot_mean_std(c1_ts,args.plot_all,color='b')
        if vf_enabled:
            plot_mean_std(delta0_ts,args.plot_all,color='m')
            plot_mean_std(delta1_ts,args.plot_all,color='y')
        plt.axhline(0,color='k')


        plt.gca().set_title("cost of two different actions")
        # plt.legend(["action 1","action 2"])
        plt.subplot(2,1,2)

        plot_mean_std(np.cumsum(c0_ts-c1_ts,axis=0),args.plot_all,color='g')
        if vf_enabled: plot_mean_std(np.cumsum(delta0_ts-delta1_ts,axis=0),args.plot_all,color='c')
        plt.axhline(0,color='k')
        for (ilam,lam) in enumerate([.8,.9,.95,.99,1]):
            if vf_enabled: 
                advest = discountedsum(delta0_ts.mean(axis=1)-delta1_ts.mean(axis=1),lam)            
                plt.axhline(advest,color='y')
                plt.annotate("%.2f"%lam, (100+10*ilam,advest) )
                
        plt.gca().set_title("accumlated cost difference")
        plt.xlabel("timestep after action")

        if args.save_prefix:
            outfile = args.save_prefix + "%.2i.pdf"%n
            plt.savefig(outfile)
        else:
            raw_input('press enter')



    # if args.burn_in_mode == "fixed":





if __name__ == "__main__":
    main()