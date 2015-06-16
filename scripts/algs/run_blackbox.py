#!/usr/bin/env python
from control4.algs.alg_params import *
from control4.config import setup_logging,print_theano_config
from control4.misc.console_utils import dict_as_table
from control4.algs.diagnostics import write_diagnostics
from control4.algs.save_load_utils import setup_outfile, save_agent_snapshot, is_save_iter
import numpy as np,os,argparse
from control4.core.rollout import rollout
from collections import defaultdict
from tabulate import tabulate

class BlackboxParams(AlgParams):
    alg = str
    paths_per_batch = int
    path_length = 9999
    cem_elite_frac = 0.1
    initial_std = 1.0
    cem_decay_time = 50
    cem_extra_std = 1.0
    policy_iter = 100
    agent_module = str
    policy_cfg = ""
    policy_kws = ""


def rollout_score(mdp,agent,max_length,th):
    agent.policy.set_var_values_flat(th)
    _,save_arrs = rollout(mdp,agent,max_length,save_arrs=["c"])
    cs = save_arrs["c"]
    return np.array(cs).sum()



def main():

    # --- BEGIN COPIED from run_spi.py ---
    # (except param_list)
    setup_logging()
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser(formatter_class=lambda prog : argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=50))
    param_list = [GeneralScriptParams,DiagnosticParams,MDPParams,BlackboxParams]
    for param in param_list:
        param.add_to_parser(parser)
    args = parser.parse_args()
    params = args.__dict__
    validate_and_filter_args(param_list, args)
    if params['test_mode']: configure_test_mode(params)
    print_theano_config()    
    print dict_as_table(params)
    # --- END COPIED from run_spi.py ---

    from control4.optim.cem import cem
    from control4.optim.cma_gen import cma_gen
    from control4.algs.save_load_utils import get_mdp,construct_agent

    mdp = get_mdp(params["mdp_name"],kws=string2dict(params["mdp_kws"]))
    agent = construct_agent(params,mdp)
    agent.pprint()
    hdf = setup_outfile(params, agent)


    f = lambda th: rollout_score(mdp,agent,params["path_length"],th)
    th = agent.policy.var_values_flat()

    if params["alg"] == "cem":
        gen = cem(f, th,batch_size=params["paths_per_batch"],n_iter=params["policy_iter"],
            elite_frac=params["cem_elite_frac"], initial_std=params["initial_std"],
            extra_std=params["cem_extra_std"],std_decay_time=params["cem_decay_time"])
    elif params["alg"] == "cma":
        gen = cma_gen(f, th,params["paths_per_batch"],params["policy_iter"],sigma=params["initial_std"])
    else:
        raise NotImplementedError
    
    diagnostics = defaultdict(list)

    for (iteration,info) in enumerate(gen):
        cs = info["ys"]
        print "Cost percentiles:"
        ps = np.linspace(0,100,11).tolist()
        print tabulate([ps, np.percentile(cs,ps)])
        diagnostics["avgcost_total"].append(info["ymean"] / params["path_length"])
        diagnostics["episodecost"].append(info["ymean"])
        assert params["path_length"] == params["diag_traj_length"]

        if is_save_iter(iteration, params["save_policy_snapshots"], params["policy_iter"]):
            save_agent_snapshot(hdf, agent, iteration)            

    write_diagnostics(hdf,diagnostics)
    hdf["runinfo"]["done"]=True


def configure_test_mode(params):
    """
    Change parameter setting so we can run this script really fast
    """
    import os.path as osp,theano #pylint: disable=F0401
    params['policy_iter'] = 2
    if "paths_per_batch" in params and params['paths_per_batch']>0: params['paths_per_batch'] = 20
    params['outfile'] = "/tmp/test_mode_output%i.h5"%os.getpid()
    if osp.exists(params['outfile']): os.unlink(params['outfile'])
    params['path_length'] = 100
    params["diag_traj_length"]= 100
    theano.config.optimizer = "None"
    import atexit
    def cleanup():
        if osp.exists(params["outfile"]):
            os.unlink(params["outfile"])
    atexit.register(cleanup)


if __name__ == "__main__":
    main()