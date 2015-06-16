#!/usr/bin/env python
import theano #pylint: disable=F0401
theano.config.floatX = 'float64'
from control4.algs.alg_params import *
from control4.config import setup_logging,print_theano_config
from control4.misc.console_utils import dict_as_table
from control4.algs.save_load_utils import setup_outfile, save_agent_snapshot, is_save_iter
from control4.algs.diagnostics import write_diagnostics
import numpy as np,os,argparse
from control4.core.rollout import rollout
from collections import defaultdict, namedtuple
from tabulate import tabulate
from control4.maths import symbolic
import theano.tensor as TT #pylint: disable=F0401

class RWRParams(AlgParams):
    paths_per_batch = int
    path_length = 9999
    elite_frac = 0.1
    policy_iter = 100
    agent_module = str
    policy_cfg = ""
    policy_kws = ""

PathData = namedtuple("PathData",["a","b","c","o"])

def rwr(mdp,agent,params):

    input_info = agent.input_info()
    output_info = agent.output_info()
    (o_size,o_dtype) = input_info['o']
    # (a_size,a_dtype) = input_info['a']
    (b_size,b_dtype) = output_info['b']
    sh_o = theano.shared(np.zeros((0,o_size),o_dtype))
    a = agent.ponder({"o":sh_o})["a"]
    sh_b = theano.shared(np.zeros((0,b_size),b_dtype))
    logliks = agent.cpd().logliks(a,sh_b)
    loss = -logliks.sum()
    th = TT.vector("th")
    policy_vars = agent.policy.vars()
    param2th = zip(policy_vars, symbolic.unflatten(th, agent.policy.var_shapes(), policy_vars)) 

    lossandgrad = theano.function([th],theano.clone([loss, symbolic.flatten(TT.grad(loss,policy_vars))],replace=param2th))

    th = agent.policy.var_values_flat()

    for _ in xrange(params["policy_iter"]):
        paths = []
        for _ in xrange(params["paths_per_batch"]):
            init_arrs,traj_arrs = rollout(mdp, agent, params['path_length'],save_arrs=("a","b","c","o"))
            a = np.concatenate(traj_arrs["a"])
            b = np.concatenate(traj_arrs["b"])
            c = np.concatenate(traj_arrs["c"])
            o = np.concatenate([init_arrs["o"]]+traj_arrs["o"][:-1])
            assert a.shape[0] == b.shape[0] == c.shape[0] == o.shape[0]
            path = PathData(a,b,c,o)
            paths.append(path)

        ctotals = np.array([path.c.sum() for path in paths])
        n_elite = int(np.round(len(paths)*params["elite_frac"]))
        elite_inds = ctotals.argsort()[:n_elite]
        elite_paths = [paths[i] for i in elite_inds]
        sh_o.set_value(np.concatenate([path.o for path in elite_paths]))
        sh_b.set_value(np.concatenate([path.b for path in elite_paths]))

        from scipy import optimize
        (th,_y,_d) = optimize.fmin_l_bfgs_b(lossandgrad,th,None,maxfun=100)
        agent.policy.set_var_values_flat(th)
        
        yield {"ys":ctotals,"ymean":ctotals.mean(),"th":th}

# Entirely copied from run_blackbox.py
def main():

    # --- BEGIN COPIED from run_spi.py ---
    # (except param_list)
    setup_logging()
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser(formatter_class=lambda prog : argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=50))
    param_list = [GeneralScriptParams,DiagnosticParams,MDPParams,RWRParams]
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


    gen = rwr(mdp,agent, params)
    
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

    hdf["runinfo"]["done"]=True
    write_diagnostics(hdf,diagnostics)

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
    params["diag_traj_length"] = 100
    theano.config.optimizer = "None"
    import atexit
    def cleanup():
        if osp.exists(params["outfile"]):
            os.unlink(params["outfile"])
    atexit.register(cleanup)


if __name__ == "__main__":
    main()