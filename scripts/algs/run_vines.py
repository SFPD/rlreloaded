#!/usr/bin/env python
from control4.algs.alg_params import *
from control4.config import setup_logging,print_theano_config
from control4.misc.console_utils import dict_as_table,Message
from control4.algs.save_load_utils import setup_outfile, save_agent_snapshot, is_save_iter
import numpy as np,os,argparse
from control4.core.rollout import rollout
from collections import defaultdict, namedtuple
from control4.maths import symbolic
import theano.tensor as TT,theano #pylint: disable=F0401
from control4.core.vines import sample_rollout_set,generate_branches,compute_branch_advantages
from control4.optim.cg_optimize import cg_optimize
from control4.algs.diagnostics import compute_path_diagnostics,write_diagnostics

class VineParams(AlgParams):
    vine_length = 100
    num_vines = 100
    branch_length = 20
    all_actions = 0
    rollout_set_size = 1000
    action_samples = 8
    trials_per_action = 1
    vine_is_method = "self_normalized"


# TODO: L2 coeff
# TODO: vine tests
# TODO kloldnew and other stats
# TODO: print out stats about num timesteps etc
# TODO: just collect the right amount of vines

PathData = namedtuple("PathData",["a","c"])

def make_path(_init_arrs,traj_arrs):
    a = np.concatenate(traj_arrs["a"])
    c = np.concatenate(traj_arrs["c"])
    path = PathData(a,c)
    return path

def compute_path_info(agent,_,paths):
    "mostly copied from pathinfomapper"
    cpd = agent.cpd()
    lengths = []
    entropies = []
    li_c_tv = []
    for path in paths:
        lengths.append(path.a.shape[0])
        entropies.append(cpd.entropy(path.a))
        li_c_tv.append(path.c)
    return {"length":lengths, "entropy":entropies, "cost":li_c_tv}


def vine_alg(mdp,agent,params):

    cpd = agent.cpd()
    
    o = TT.matrix("o",agent.input_dtype("o"))    
    olda_nk = TT.matrix("olda",agent.output_dtype("a"))
    b_nkb = TT.tensor3("b",agent.output_dtype("b")) 
    th = TT.vector("th")
    dth = TT.vector("dth")
    adv_nk = TT.matrix("adv")
    q_nk = TT.matrix("q")
    policy_vars = agent.policy.vars()


    newa_na = agent.ponder({"o":o})["a"]
    N = b_nkb.shape[0]
    K = b_nkb.shape[1]
    B = b_nkb.shape[2]
    p_nk = cpd.liks(TT.repeat(newa_na,K,axis=0),b_nkb.reshape([N*K,B])).reshape([N,K])
    if params["vine_is_method"] == "self_normalized":
        w_nk = p_nk/q_nk
        w_nk = w_nk / w_nk.sum(axis=1,keepdims=True)
        loss = (w_nk*adv_nk).sum(axis=1).mean()
    elif params["vine_is_method"] == "mean_baseline":
        w_nk = p_nk/q_nk
        dmadv_nk = adv_nk - adv_nk.mean(axis=1,keepdims=True)
        loss = (w_nk*dmadv_nk).sum(axis=1).mean()
    else:
        raise NotImplementedError
    dparams = symbolic.unflatten(dth, agent.policy.var_shapes(), agent.policy.vars())
    param2th = zip(policy_vars, symbolic.unflatten(th, agent.policy.var_shapes(), policy_vars)) 

    kl = cpd.kl(olda_nk,newa_na).mean()
    losses = [loss,kl]
    loss_names = ["loss","kl"]

    flosses = theano.function([th,adv_nk,olda_nk, b_nkb,q_nk,o],theano.clone(losses,replace=param2th))
    fgradloss = theano.function([th,adv_nk,b_nkb,q_nk,o],symbolic.flatten(theano.clone(TT.grad(loss,agent.policy.vars()),replace=param2th)))
    fvp = TT.Lop(newa_na, policy_vars, cpd.fvp(newa_na,TT.Rop(newa_na, policy_vars, dparams)))
    flatfvp = symbolic.flatten(fvp)/N
    f_fvp = theano.function( [th, dth,adv_nk,b_nkb,q_nk,o],theano.clone(flatfvp,replace=param2th),
        on_unused_input='ignore',allow_input_downcast=True)


    b = TT.matrix("b",agent.output_dtype("b"))
    fs_b2u = theano.function([b],agent.b2u(b))

    th = agent.policy.var_values_flat()

    for _ in xrange(params["policy_iter"]):

        paths = []
        inittrajs = []
        with Message("making vines"):            
            for _ in xrange(params["num_vines"]):
                inittraj = (init,traj) = rollout(mdp,agent,params["vine_length"],save_arrs=("x","o","a","c"))
                paths.append(make_path(init,traj))
                inittrajs.append(inittraj)

        rsd = sample_rollout_set(inittrajs,params["rollout_set_size"])
        with Message("making branches"):
            bd = generate_branches(mdp, agent, rsd,
                branch_length = params["branch_length"], 
                actions_per_state = params["action_samples"],
                trials_per_action = params["trials_per_action"], 
                all_actions = params["all_actions"],
                fs_b2u = fs_b2u
            )

        adv = compute_branch_advantages(bd, params["gamma"],params["lam"])
        if params["standardize_adv"]:
            adv -= adv.mean(axis=1)[:,None]
            adv /= adv.std(axis=1).mean()

        def pen_loss(*args):
            loss,kl = flosses(*args)
            return loss + 1e10*(kl > 2*params['pol_cg_kl'])

        th_after = cg_optimize(th,
            lambda th: pen_loss(th, adv,rsd.a_na, bd.b_nkb,bd.sq_nk,rsd.o_no), #pylint: disable=W0640
            lambda th: fgradloss(th, adv,bd.b_nkb,bd.sq_nk,rsd.o_no), #pylint: disable=W0640
            params['pol_cg_kl'], 1, params['pol_cg_damping'],
            cg_iters=params['pol_cg_iters'], 
            do_linesearch=True,
            min_lm=params["pol_cg_min_lm"],
            fmetric=lambda th,p: f_fvp(th,p,adv,bd.b_nkb,bd.sq_nk,rsd.o_no)) #pylint: disable=W0640

        losses_before = np.array(flosses(th,adv,rsd.a_na,bd.b_nkb,bd.sq_nk,rsd.o_no))
        losses_after = np.array(flosses(th_after,adv,rsd.a_na,bd.b_nkb,bd.sq_nk,rsd.o_no))
        delta_losses = losses_before - losses_after
        from tabulate import tabulate
        print tabulate(zip(loss_names, losses_before, losses_after,delta_losses), headers=["loss","before","after","total"])



        th = th_after
        agent.policy.set_var_values_flat(th)

        # vine_costs = np.array([np.array(traj["c"]).sum() for (_,traj) in inittrajs])
        yield {"paths":paths}

# Entirely copied from run_blackbox.py
def main():

    # --- BEGIN COPIED from run_spi.py ---
    # (except param_list)
    setup_logging()
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser(formatter_class=lambda prog : argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=50))
    param_list = [GeneralScriptParams,DiagnosticParams,MDPParams,VineParams,SPIParams,PolicyOptParams]
    for param in param_list:
        param.add_to_parser(parser)
    args = parser.parse_args()
    params = args.__dict__
    validate_and_filter_args(param_list, args)
    if params['test_mode']: configure_test_mode(params)
    print_theano_config()    
    print dict_as_table(params)
    # --- END COPIED from run_spi.py ---

    from control4.algs.save_load_utils import get_mdp,construct_agent

    mdp = get_mdp(params["mdp_name"],kws=string2dict(params["mdp_kws"]))
    agent = construct_agent(params,mdp)
    agent.pprint()
    hdf = setup_outfile(params, agent)


    gen = vine_alg(mdp,agent, params)
    
    path_diags = defaultdict(list)

    for (iteration,info) in enumerate(gen):
        print "***** Iteration %i *****"%iteration
        path_info = compute_path_info(agent,mdp,info["paths"])
        compute_path_diagnostics(mdp, path_diags, path_info, params["diag_traj_length"])
        if is_save_iter(iteration, params["save_policy_snapshots"], params["policy_iter"]):
            save_agent_snapshot(hdf, agent, iteration)            
            
    write_diagnostics(hdf, path_diags)
    hdf["runinfo"]["done"]=True


def configure_test_mode(params):
    """
    Change parameter setting so we can run this script really fast
    """
    import os.path as osp
    params['policy_iter'] = 2
    if "paths_per_batch" in params and params['paths_per_batch']>0: params['paths_per_batch'] = 20
    params['outfile'] = "/tmp/test_mode_output%i.h5"%os.getpid()
    if osp.exists(params['outfile']): os.unlink(params['outfile'])
    params['path_length'] = 100
    theano.config.optimizer = "None"
    import atexit
    def cleanup():
        if osp.exists(params["outfile"]):
            os.unlink(params["outfile"])
    atexit.register(cleanup)


if __name__ == "__main__":
    main()