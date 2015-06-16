import itertools,sys,numpy as np,h5py,os,logging,time,os.path as osp
from tabulate import tabulate
import theano,theano.tensor as TT #pylint: disable=F0401
from control4.config import floatX
from control4.misc.func_utils import sum_count_reducer
from control4.misc.console_utils import Message,Timers
from control4.core.rollout import rollout
from control4.maths.numeric import explained_variance_1d
from control4.maths import symbolic
from control4.optim.cg_optimize import cg_optimize,linesearch
from control4.maths.discount import discount
from control4.misc.collection_utils import concatenate
from control4.algs.advantage_est import compute_baseline
from control4.algs.diagnostics import compute_path_diagnostics,write_diagnostics
from control4.algs.alg_params import string2dict
from control4.algs.save_load_utils import construct_agent,fetch_file,\
    setup_outfile,get_mdp,save_agent_snapshot,is_save_iter
from control4.parallel.parallel import DummyPool,ProcessPool
from collections import namedtuple,defaultdict,OrderedDict
################################
# Global stuff
################################

class G: #pylint: disable=W0232
    # Misc globals
    mdp = None
    params = None
    agent = None
    pool = None
    vvar = theano.shared(np.array(0,floatX),name='vvar')

    # Updated every iteration
    advs = []
    paths = []
    vtargs = []

    seeds = []

class Glf: #pylint: disable=W0232
    # global loss functions
    pass

################################


def get_local_paths(sli):
    if isinstance(sli,int):
        return [G.paths[sli]]
    elif isinstance(sli, np.ndarray):
        return [G.paths[i] for i in sli]
    elif sli == "train":
        return (path for (i,path) in itertools.izip(itertools.count(),G.paths) if i%4!=1)
    elif sli == "test":
        return (path for (i,path) in itertools.izip(itertools.count(),G.paths)  if i%4==1)
    elif sli == "all":
        assert G.paths is not None
        return G.paths
    else:
        raise RuntimeError("invalid slice specification %s"%sli)

def get_local_advpathvtargs(sli):
    if isinstance(sli,int):
        return [(G.advs[sli],G.paths[sli],G.vtargs[sli])]
    elif isinstance(sli, np.ndarray):
        return [(G.advs[i],G.paths[i],G.vtargs[i]) for i in sli]
    elif sli == "train":
        return ((adv,path,vtarg) for (i,adv,path,vtarg) in itertools.izip(itertools.count(),G.advs,G.paths,G.vtargs) if i%4!=1)
    elif sli == "test":
        return ((adv,path,vtarg) for (i,adv,path,vtarg) in itertools.izip(itertools.count(),G.advs,G.paths,G.vtargs) if i%4==1)
    elif sli == "all":
        return itertools.izip(G.advs, G.paths,G.vtargs)
    else:
        raise RuntimeError("invalid slice specification %s"%sli) 

AVERAGER_FUNCS = []
class AveragerFuncWrapper(object):
    """
    Wrap an un-pickleable function here, and its associated with a unique key
    Make sure that remote processes or child processes make the same sequence of calls to this constructor
    so they have the appropriate functions
    """
    def __init__(self, func):
        self.key = len(AVERAGER_FUNCS)
        AVERAGER_FUNCS.append(func)
    def __call__(self,*args):
        return AVERAGER_FUNCS[self.key](*args)
class Averager(object):
    def __init__(self, func):
        self.mapper = AveragerFuncWrapper(func)
    def __call__(self,*args):
        total,count= G.pool.gather(
            self.mapper,
            sum_count_reducer,
            args
        )
        return total/count

################################

PathData = namedtuple("PathData", ["o", "c", "initm", "b", "a", "q", "v", "done", "T", "ro"])

# o: observation
# c: cost
# a: distribution parameter
# b: sampled value
# q: p(b|a)
# v: cost-to-go
# done: done at that final timestep?
# T: number of timesteps completed 
#   = number of times policy and dynamics called. note that we have T+1 of some vars

def path_from_rollout(init,traj,Tmax):
    o = np.concatenate([init["o"]]+traj["o"][:Tmax])
    c = np.concatenate(traj["c"])
    initm = init.get("m")
    b = np.concatenate(traj["b"])[:Tmax]
    a = np.concatenate(traj["a"])[:Tmax]
    q = np.concatenate(traj["q"])[:Tmax]
    T = o.shape[0]-1
    if "v" in traj:
        traj["v"].append([0] if traj['done'][-1] else traj['v'][-1]) # XXX last value is probably approximately correct
        v = np.concatenate(traj["v"])
    else:
        v = np.zeros(c.shape[0]+1,floatX)
    done = traj["done"][-1] if "done" in traj else False
    ro = np.array(traj["ro"])[:Tmax] if "ro" in traj else None
    return PathData(o,c,initm,b,a,q,v,done,T,ro)

def generate_and_store_paths(li_seed):

    save_arrs = set(["o","c","done","ro",  "a","b","q","m","v"])
    save_arrs.intersection_update(G.agent.output_info().keys()+G.mdp.output_info().keys())
    save_arrs = list(save_arrs)

    path_lengths = []
    max_length = G.params['path_length'] + int(G.params["path_extension_factor"]*np.ceil(1.0/(1.0-G.params['gamma'])) if "path_extension_factor" in G.params else 0)
    for seed in li_seed:
        np.random.seed(seed)
        init_arrs,traj_arrs = rollout(G.mdp, G.agent, max_length,save_arrs=save_arrs) 
        path = path_from_rollout(init_arrs,traj_arrs,G.params["path_length"])
        G.paths.append(path)
        G.seeds.append(seed)
        path_lengths.append(path.T)

    return path_lengths


def set_policy_var_values_flat(th):
    G.agent.policy.set_var_values_flat(th)

def set_vf_var_values_flat(th):
    G.agent.vf.set_var_values_flat(th)

# For linear vaue function there are also integer parameters (good indices)
def set_vf_var_values(vals):
    G.agent.vf.set_var_values(vals)

def clear_paths(_):
    G.paths = []
    G.advs = []
    G.vtargs = []

def get_cost_value_mapper():
    return [ (path.c.sum(axis=1), path.v) for path in get_local_paths("all") ]

def set_vvar(v):
    G.vvar.set_value(v)    

def get_paths_mapper(_):
    return G.paths


def merge_dicts(ds):
    """
    list of dict of list -> dict of list
    """
    dtotal = {key:[] for key in ds[0]}
    for d in ds:
        for (key,val) in d.iteritems():
            dtotal[key].extend(val)
    return dtotal

def pathinfomapper(_):
    cpd = G.agent.cpd()
    lengths = []
    entropies = []
    li_c_tv = []
    for path in get_local_paths("all"):
        lengths.append(path.T)
        entropies.append(cpd.entropy(path.a))
        li_c_tv.append(path.c)
    return {"length":lengths, "entropy":entropies, "cost":li_c_tv}

def compute_and_store_advantages(_):

    gamma = G.params['gamma']
    lam = G.params['lam']
    vf_lam = G.params['vf_lam']

    if G.params["deepmind_cheat_mode"]:
        # XXX NOT PROPERLY USING VALUE FUNCTION
        # TODO: ASSERT NO VALUE FUNCTION BEING USED
        li_adv = []
        li_delta_t = []
        for path in get_local_paths("all"):
            tstart = 0
            adv_pieces = []
            path.ro[-1] = True
            for tnextstart in np.flatnonzero(path.ro)+1:
                adv_pieces.append(discount(path.c[tstart:tnextstart].sum(axis=1),gamma))
                tstart=tnextstart
            adv = np.concatenate(adv_pieces)
            assert len(adv) == len(path.c)
            li_adv.append(adv)
            li_delta_t.append(path.c.sum(axis=1))

    vtargs = []
    li_delta_t = []
    li_adv = []
    for path in G.paths:
        end_mode = G.params["vf_end_mode"]
        if end_mode == "zero":
            path.v[path.T] = 0
        elif end_mode == "bootstrap":
            if path.done:            
                path.v[path.T] = 0
        else:
            raise NotImplementedError

        delta_t = path.c.sum(axis=1)+gamma*path.v[1:] - path.v[0:-1]
        vtarg = path.v[0:path.T] + discount(delta_t, gamma*vf_lam)[0:path.T]
        # if vf_lam == 1:
        #     assert np.allclose(vtarg, discount(np.concatenate([path.c.sum(axis=1),[0*path.v[-1]]]), gamma)[-1])
        adv = discount(delta_t, gamma*lam)[0:path.T]
        li_adv.append(adv)
        vtargs.append(vtarg)
    
    G.advs = li_adv
    G.vtargs = vtargs

    vcurs = [path.v[0:path.T] for path in G.paths]
    return zip(G.advs, G.vtargs, vcurs)

def shift_and_scale_advantages((baseline,scaling)):
    for adv in G.advs:
        if baseline is not None: adv -= baseline[:len(adv)]
        adv /= scaling


################################


def generate_paths_toplevel(params, seed_iter):
    # Do actual generation on slave processes/computers
    G.pool.apply(set_policy_var_values_flat, G.agent.policy.var_values_flat())
    G.pool.apply(set_vf_var_values_flat, G.agent.vf.var_values_flat())
    G.pool.apply(clear_paths,None)


    if params['paths_per_batch'] != 0:
        # x0_nd = iss.get_n(params['paths_per_batch'])
        li_seed = list(itertools.islice(seed_iter,params['paths_per_batch']))
        n_paths = params['paths_per_batch']
        path_lengths = concatenate(G.pool.scatter(generate_and_store_paths, li_seed))
        n_timesteps = sum(path_lengths)
    elif params['timesteps_per_batch'] != 0:
        n_timesteps = 0
        n_paths = 0
        while n_timesteps < params['timesteps_per_batch']:
            li_seed = list(itertools.islice(seed_iter,params['path_chunk_size']))
            path_lengths = concatenate(G.pool.scatter(generate_and_store_paths, li_seed))
            n_paths += len(path_lengths)
            n_timesteps += sum(path_lengths)
    else:
        raise NotImplementedError

    print "got %i paths with total num timesteps %i"%(n_paths,n_timesteps)


### Should compute baseline using ALL paths


def create_non_cluster_pool(params):
    if params['par_mode'] == "off" or params['n_processes'] == 1:
        return DummyPool()
    elif params['par_mode'] == "multiprocessing":
        if sys.platform == "darwin":
            return ProcessPool(params['n_processes'])
        else:
            from control4.parallel.ipcprocesspool import IPCProcessPool
            return IPCProcessPool(params['n_processes'])
    else:
        raise NotImplementedError    

def create_cluster_pool(params):
    from control4.cloud.cluster_pool import ClusterPool
    from control4.cloud.cloud_interface import create_cloud,load_cloud_config
    cloud_config = load_cloud_config(provider=params['cloud_provider'])
    cloud = create_cloud(cloud_config)
    return ClusterPool(cloud,params['cluster_name'],start_mode="the_prestige")

####################
# Optimization helpers 
####################
 
def make_loss_funcs_for_unc_batch_policy_opt(params,mdp,agent): #pylint: disable=W0613
    """
    Makes loss functions for optimizing squashed or penalized version of expected cost objective, using full-batch method (LBFGS)
    Generates the following functions:
    - f_losses :: th:[float],train_or_test:str => [float]
    - f_trainloss :: th:[float] => float
    - f_gradtrainloss :: th:[float] => [float]
    - f_testloss :: th:[float] => float
    """
    raise NotImplementedError    


def make_loss_funcs_for_ipm_policy_opt(params,mdp,agent): #pylint: disable=W0613
    """
    Makes loss functions for optimizing expected cost objective subject to a constraint using 
    - f_losses :: th:[float],train_or_test:str => dict
    - f_trainlag :: th:[float],lam:float => float
    - f_gradtrainlag :: th:[float],lam:float => [float]
    - f_testloss :: th:[float] => float
    Lagrangians for interior point method
    """
    raise NotImplementedError

def addsq(f):
    def f1(*args):
        s,c = f(*args)
        return (np.array([s,np.square(s)]),c)
    return f1

# ["prevo", "prevc", "prevm", "prevb", "a", "q", "v", "done", "T"]
# ["prevo", "prevc", "prevm", "prevb", "a", "q", "v", "done", "T"]
def make_loss_funcs_for_cg_policy_opt(params,mdp,agent): #pylint: disable=W0613
    """
    Make loss functions for optimizing policy by CG
    - f_losses :: th:[float],train_or_test:str,th_old:[float] => dict
    - f_trainloss :: th:[float],th_old:[float] => float
    - f_gradtrainloss :: th,th_old:[float] => [float]
    - f_fvp :: th:[float],p:[float],th_old:[float] => Hp:[float]
    - f_testloss :: th:[float],th_old:[float] => float
    """

    rnn_mode = "m" in agent.input_info()

    # Parameter arrays
    th = TT.vector("th")
    # th_old = TT.vector("th_old")
    dth = TT.vector('dth')

    # agent_inputs = agent.input_info().keys()
    # agent_outputs = agent.output_info().keys()
    # assert len(set(agent_inputs).intersection(agent_outputs))==0 # Otherwise we'll have key/name collision

    ################################
    # Main arrs includes the inputs and outputs from the policy
    main_arrs = OrderedDict([
        ("o",TT.matrix("o",agent.input_dtype("o"))),
        ("b",TT.matrix("b",agent.output_dtype("b"))),
        ("a",TT.matrix("a",floatX)),
        ("q",TT.vector("q",floatX)),
        ("adv",TT.vector("adv",floatX))
    ])
    N = main_arrs["a"].shape[0]

    ################################
    # pd_arrs will be associated with the arrays from the PathData struct
    pd_arrs = OrderedDict([(name,theano.clone(arr)) for (name,arr) in main_arrs.iteritems()])
    T = pd_arrs["a"].shape[0]

    loss_map = [(main_arrs[name],pd_arrs[name][0:T]) for name in pd_arrs]    
    hess_subsample = 1 if rnn_mode else params["pol_hess_subsample"]    
    fish_map = [(main_arrs[name],pd_arrs[name][0:T:hess_subsample]) for name in pd_arrs]    

    ################################

    if rnn_mode:
        pd_arrs["initm"] = initm = TT.matrix("initm")
        def onestep(o,m):
            d = agent.ponder({"o":o[None,:],"m":m[None,:]})
            m = d["m"][0]
            a = d["a"][0]
            return m,a
        (main_arrs["m"],newa_na),_ = theano.scan(fn=onestep, sequences=[main_arrs["o"]], outputs_info=[dict(initial=initm[0],taps=[-1]),dict()],n_steps=N)


    loss_input_names = [name for name in PathData._fields if name in pd_arrs] #pylint: disable=W0212,E1101
    loss_inputs = [pd_arrs[name] for name in loss_input_names]
    loss_inputs_from_path = lambda path: [getattr(path,key) for key in loss_input_names]


    # OK, now let's construct the loss function
    policy_vars = agent.policy_vars()    
    cpd = agent.cpd()

    if not rnn_mode:
        newa_na = agent.ponder(main_arrs)["a"]

    p_n = cpd.liks(newa_na,main_arrs["b"])
    q_n = main_arrs["q"]    
    w_n = p_n/q_n
    dparams = symbolic.unflatten(dth, agent.policy.var_shapes(), agent.policy.vars())

    param2th = zip(policy_vars, symbolic.unflatten(th, agent.policy.var_shapes(), agent.policy.vars())) 


    ec = w_n.dot(main_arrs["adv"])
    allkloldnew = cpd.kl(main_arrs["a"], newa_na)
    assert allkloldnew.ndim == 1
    kloldnewsum =  allkloldnew.sum()

    ent = - cpd.entropy(newa_na).sum() * params["pol_ent_coeff"]

    l2 = N*params['pol_l2_coeff']*agent.policy.l2()
    losses = [ec,ent,kloldnewsum,l2,ec+l2+ent]
    loss_names = ["ec","ent","kl","l2","tot"]
    total_train_idx = total_test_idx = loss_names.index("tot")
    kl_idx = loss_names.index("kl")
    total_train_loss = losses[total_train_idx]

    if 0:
        gradkl_terms = TT.grad(kloldnewsum,policy_vars)
        gdotp = TT.add(*[(g*dp).sum() for (g,dp) in zip(gradkl_terms,dparams)]) #pylint: disable=E1111
        hps = TT.grad(gdotp, policy_vars)
        fvpnomu = theano.clone(hps, replace={main_arrs["a"]:newa_na})
        flatfvpnomu = symbolic.flatten(fvpnomu)
        f_fvp = theano.function(
            [th, dth] + loss_inputs,
            theano.clone([flatfvpnomu,N],replace=fish_map+param2th),
            on_unused_input='ignore',allow_input_downcast=True)
    else:
        fvp = TT.Lop(newa_na, policy_vars, cpd.fvp(newa_na,TT.Rop(newa_na, policy_vars, dparams)))
        flatfvpnomu = symbolic.flatten(fvp)
        f_fvp = theano.function(
            [th, dth] + loss_inputs,
            theano.clone([flatfvpnomu,N],replace=fish_map+param2th),
            on_unused_input='ignore',allow_input_downcast=True)


    floss = theano.function(
        # [th, adv_t, prevm_tg, prevj_tb, o_tf, sq_t, oldmu_tk], 
        [th,pd_arrs["adv"]] + loss_inputs,
        [TT.stack(*theano.clone(losses, replace=loss_map+param2th)),N], 
        on_unused_input='ignore',allow_input_downcast=True)
    gradfloss = theano.function(
        [th,pd_arrs["adv"]] + loss_inputs,
        # [th, adv_t, prevm_tg, prevj_tb, o_tf, sq_t, oldmu_tk], 
        [theano.clone(symbolic.flatten(TT.grad(total_train_loss,policy_vars)), replace=loss_map+param2th),N], 
        on_unused_input='ignore',allow_input_downcast=True)

    if params['fancy_damping']:
        gradfloss = addsq(gradfloss)

    if params['pol_opt_alg'] == 'maxkl':
        J_na = TT.Rop(newa_na, policy_vars, dparams)
        MJ_na = cpd.fvp(newa_na, J_na)
        f_hess_stuff = theano.function(
            [th,dth] + loss_inputs,
            theano.clone([J_na, MJ_na],replace=loss_map+param2th),
            on_unused_input='ignore',allow_input_downcast=True)
        def hessians(th,*args):
            # just to get size
            J_na, _ = f_hess_stuff(th,th*0,*args)
            J_qna = np.zeros((th.size, J_na.shape[0], J_na.shape[1]), floatX)
            MJ_qna = np.zeros((th.size, J_na.shape[0],J_na.shape[1]), floatX)
            for q in xrange(th.size):                
                dth = np.zeros_like(th)
                dth[q] = 1
                J_qna[q], MJ_qna[q] = f_hess_stuff(th,dth,*args)
            JMJ_naa = np.einsum("Qna,qna->nQq",J_qna, MJ_qna)
            return JMJ_naa
        f_allkl = theano.function([th]+loss_inputs,theano.clone(allkloldnew,replace=loss_map+param2th), on_unused_input='ignore',allow_input_downcast=True)        
        Glf.f_hessians = staticmethod(lambda th: np.concatenate([hessians(th,*loss_inputs_from_path(path)) for path in get_local_paths("train")],axis=0))
        Glf.f_allkl = staticmethod(lambda th: np.concatenate([f_allkl(th,*loss_inputs_from_path(path)) for path in get_local_paths("train")],axis=0))


    G.pol_loss_names = loss_names

    meanlosses = Averager(lambda (th,sli): sum_count_reducer(tuple(floss(th, adv, *loss_inputs_from_path(path))) for (adv,path,vtarg) in get_local_advpathvtargs(sli)))
    meangradloss = Averager(lambda (th,sli): sum_count_reducer(tuple(gradfloss(th, adv, *loss_inputs_from_path(path))) for (adv,path,vtarg) in get_local_advpathvtargs(sli)))
    if rnn_mode:
        hess_subsample = params["pol_hess_subsample"]
        meanfvp = Averager(lambda (th,p,sli): sum_count_reducer(tuple(f_fvp(th,p,*loss_inputs_from_path(path))) for path in itertools.islice(get_local_paths(sli),0,None,hess_subsample)))
    else:
        meanfvp = Averager(lambda (th,p,sli): sum_count_reducer(tuple(f_fvp(th,p,*loss_inputs_from_path(path))) for path in get_local_paths(sli)))

    def samptrainloss(th):
        "add barrier for violating kl constraint"
        samplosses = meanlosses(th,"train")
        return samplosses[total_train_idx] + 1e10*(samplosses[kl_idx] > 2*params['pol_cg_kl'])


    Glf.f_losses = staticmethod(lambda th,sli,th_old: meanlosses(th,sli))
    # Unpenalized version:
    # Glf.f_trainloss = staticmethod(lambda th,th_old: meanlosses(th,"train")[total_train_idx])
    Glf.f_trainloss = staticmethod(lambda th,th_old: samptrainloss(th))
    Glf.f_gradtrainloss = staticmethod(lambda th,th_old: meangradloss(th,"train"))
    Glf.f_testloss = staticmethod(lambda th,th_old: meanlosses(th,"test")[total_test_idx])
    Glf.f_trainfvp = staticmethod(lambda th,p,th_old: meanfvp(th,p,"train"))

def make_gradlogps(mdp,agent):
    o = TT.matrix("o",mdp.output_dtype("o"))
    b = TT.matrix("b",agent.output_dtype("b"))
    newa = agent.ponder({"o":o})["a"]
    logp_n = agent.cpd().logliks(newa, b)
    def onegrad(i):
        logp1 = theano.clone(logp_n, replace = {b:b[i:i+1],o:o[i:i+1]})[0]
        return symbolic.flatten(TT.grad(logp1, agent.policy_vars()))
    gradlogps,_ = theano.map(onegrad, TT.arange(logp_n.shape[0]))
    Glf.ftheano_gradlogp = theano.function([o,b],gradlogps)
    Glf.f_gradlogp = staticmethod(lambda : G.pool.gather(gradlogpmapper,np.concatenate,None))

def gradlogpmapper(_):
    return np.concatenate([Glf.ftheano_gradlogp(path.o[:-1],path.b) for path in get_local_paths("train")])

def make_loss_funcs_for_cg_vf_opt(params,mdp,agent): #pylint: disable=W0613
    """
    Make loss functions for optimizing value function by CG
    - f_vflosses :: th:[float],train_or_test:str,th_old:[float] => dict
    - f_vftrainloss :: th:[float],th_old:[float] => float
    - f_vfgradtrainloss :: th,th_old:[float] => [float]
    - f_vftrainfvp :: th:[float],p:[float],th_old:[float] => Hp:[float]
    - f_vftestloss :: th:[float],th_old:[float] => float
    """

    rnn_mode = "m" in agent.input_info()

    # Parameter arrays
    th = TT.vector("th")
    # th_old = TT.vector("th_old")
    dth = TT.vector('dth')

    ################################
    # Main arrs includes the inputs and outputs from the policy
    main_arrs = OrderedDict([
        ("o",TT.matrix("o",agent.input_dtype("o"))),
        ("vcur",TT.vector("vcur",floatX)),
        ("vtarg",TT.vector("vtarg",floatX)),
    ])
    vcur = main_arrs["vcur"]
    vtarg = main_arrs["vtarg"]
    N = vtarg.shape[0]

    ################################
    # pd_arrs will be associated with the arrays from the PathData struct
    pd_arrs = OrderedDict([(name,theano.clone(arr)) for (name,arr) in main_arrs.iteritems()])
    T = pd_arrs["vtarg"].shape[0]

    loss_map = [(main_arrs[name],pd_arrs[name][0:T]) for name in pd_arrs]
    hess_subsample = 1 if rnn_mode else params["vf_hess_subsample"]    
    fish_map = [(main_arrs[name],pd_arrs[name][0:T:hess_subsample]) for name in pd_arrs]    

    ################################

    if rnn_mode:
        raise NotImplementedError


    # loss_map.append((main_arrs["b"],pd_arrs["b"][1:T+1]))

    loss_input_names = ["o","vcur","vtarg"] #pylint: disable=W0212,E1101
    loss_inputs = [pd_arrs[name] for name in loss_input_names]

    # fish_map.append((main_arrs["b"],pd_arrs["b"][1:T+1:hess_subsample]))
    # fish_map[main_arrs["b"]] = pd_arrs["prevb"][1:T+1]
    # fish_inputs = pd_arrs.values() MINUS adv

    # add param2th

    # OK, now let's construct the loss function
    vf_vars = agent.vf_vars()    

    if not rnn_mode:
        vpred = agent.ponder(main_arrs)["v"]

    dparams = symbolic.unflatten(dth, agent.vf.var_shapes(), agent.vf.vars())

    param2th = zip(vf_vars, symbolic.unflatten(th, agent.vf.var_shapes(), agent.vf.vars())) 

    assert vpred.ndim == 1 
    assert vtarg.ndim == 1
    vdiff = vpred - vtarg
    kloldnewsum = TT.square(vpred-vcur).sum() / (2*G.vvar)

    l2 = N*params['vf_l2_coeff']*agent.vf.l2()

    if params['vf_loss_type'] == "l2":
        vferr = TT.square(vdiff).sum() / (2*G.vvar)
    elif params['vf_loss_type'] == "l1":
        vferr = TT.abs_(vdiff).sum() / TT.sqrt(G.vvar)
    elif params['vf_loss_type'] == "quantile":
        vferr = (vdiff * (TT.sgn(vdiff) + (.5 - params['vf_loss_quantile'])*2)).mean()
        # quantile=0 -> only penalize positive 
        # quantile=1 -> only penalize negative
    elif params['vf_loss_type'] == "l2_asym":
        vferr = TT.square(vdiff * (TT.sgn(vdiff) + (.5 - params['vf_loss_quantile'])*2)).mean()
        # quantile=0 -> only penalize positive 
        # quantile=1 -> only penalize negative
    else:
        raise NotImplementedError

    losses = [vferr,kloldnewsum,l2,vferr+l2]
    loss_names = ["vferr","kl","l2","tot"]
    total_train_idx = total_test_idx = loss_names.index("tot")
    kl_idx = loss_names.index("kl")
    total_train_loss = losses[total_train_idx]

    fvp = TT.Lop(vdiff, vf_vars, TT.Rop(vdiff / G.vvar, vf_vars, dparams))
    flatfvpnomu = symbolic.flatten(fvp)
    f_fvp = theano.function(
        [th, dth] + loss_inputs,
        theano.clone([flatfvpnomu,N],replace=fish_map+param2th),
        on_unused_input='ignore',allow_input_downcast=True)


    floss = theano.function(
        # [th, adv_t, prevm_tg, prevj_tb, o_tf, sq_t, oldmu_tk], 
        [th] + loss_inputs,
        [TT.stack(*theano.clone(losses, replace=loss_map+param2th)),N], 
        on_unused_input='ignore',allow_input_downcast=True)
    gradfloss = theano.function(
        [th] + loss_inputs,
        # [th, adv_t, prevm_tg, prevj_tb, o_tf, sq_t, oldmu_tk], 
        [theano.clone(symbolic.flatten(TT.grad(total_train_loss,vf_vars)), replace=loss_map+param2th),N], 
        on_unused_input='ignore',allow_input_downcast=True)

    if params['fancy_damping']:
        gradfloss = addsq(gradfloss)

    G.vf_loss_names = loss_names

    meanlosses = Averager(lambda (th,sli): sum_count_reducer(tuple(floss(th, path.o, path.v, vtarg)) for (_,path,vtarg) in get_local_advpathvtargs(sli)))
    meangradloss = Averager(lambda (th,sli): sum_count_reducer(tuple(gradfloss(th, path.o, path.v, vtarg)) for (_,path,vtarg) in get_local_advpathvtargs(sli)))
    if rnn_mode:
        hess_subsample = params["vf_hess_subsample"]
        raise NotImplementedError
        # meanfvp = Averager(lambda (th,p,sli): sum_count_reducer(tuple(f_fvp(th, p, path.o, path.v[:path.T], vtarg)) for path in itertools.islice(get_local_paths(sli),0,None,hess_subsample)))
    else:
        meanfvp = Averager(lambda (th,p,sli): sum_count_reducer(tuple(f_fvp(th, p, path.o, path.v[:path.T], vtarg)) for (_,path,vtarg) in get_local_advpathvtargs(sli)))
        # XXX fvp doesn't need to be a function of vtarg

    def samptrainloss(th):
        "add barrier for violating kl constraint"
        samplosses = meanlosses(th,"train")
        return samplosses[total_train_idx] + 1e10*(samplosses[kl_idx] > 2*params['vf_cg_kl'])


    Glf.f_vflosses = staticmethod(lambda th,sli,th_old: meanlosses(th,sli))
    # Unpenalized version:
    # Glf.f_trainloss = staticmethod(lambda th,th_old: meanlosses(th,"train")[total_train_idx])
    Glf.f_vftrainloss = staticmethod(lambda th,th_old: samptrainloss(th))
    Glf.f_vfgradtrainloss = staticmethod(lambda th,th_old: meangradloss(th,"train"))
    Glf.f_vftestloss = staticmethod(lambda th,th_old: meanlosses(th,"test")[total_test_idx])
    Glf.f_vftrainfvp = staticmethod(lambda th,p,th_old: meanfvp(th,p,"train"))


def make_loss_funcs_for_policy_opt(params,mdp,agent):
    if params['pol_opt_alg'] in ("cg","cg_fixed_lm","empirical_fim","maxkl"): # FOR ICML
        make_loss_funcs_for_cg_policy_opt(params,mdp,agent)
        if params['pol_opt_alg'] == "empirical_fim": make_gradlogps(mdp,agent) # FOR ICML
    else:
        raise NotImplementedError

def make_loss_funcs_for_vf_opt(params,mdp,agent):
    if params['vf_opt_alg'] == "cg":
        make_loss_funcs_for_cg_vf_opt(params,mdp,agent)
    else:
        raise NotImplementedError

def make_loss_funcs_for_joint_opt(params,mdp,agent): #pylint: disable=W0613
    raise NotImplementedError

def make_loss_funcs(params,mdp,agent):
    if params['vf_opt_mode'] in ("off","linear"):
        make_loss_funcs_for_policy_opt(params,mdp,agent)
    elif params['vf_opt_mode'] == "joint":
        make_loss_funcs_for_joint_opt(params,mdp,agent)
    elif params['vf_opt_mode'] == "separate":
        make_loss_funcs_for_policy_opt(params,mdp,agent)
        make_loss_funcs_for_vf_opt(params,mdp,agent)
    else:
        raise NotImplementedError

def get_num_paths(_):
    return len(G.paths)

def optimize_policy(params, agent, diags):

    log = logging.getLogger("optimize_policy")
    log.info("Optimizing policy")

    th_before = agent.policy.var_values_flat()

    timers = Timers()

    train_losses_before = Glf.f_losses(th_before,"train",th_before)
    test_losses_before = 0*train_losses_before if params['disable_test'] else Glf.f_losses(th_before,"test",th_before)

    if params['pol_opt_alg']=="cg":
        th_after = cg_optimize(th_before, timers.wrap(lambda th: Glf.f_trainloss(th, th_before),"trainloss"), 
                    timers.wrap(lambda th: Glf.f_gradtrainloss(th, th_before),"gradtrainloss"),
                    params['pol_cg_kl'], params['pol_cg_steps'], params['pol_cg_damping'],
                    cg_iters=params['pol_cg_iters'], fmetric=timers.wrap(lambda th,p: Glf.f_trainfvp(th,p,th_before),"fvp"), 
                    fancy_damping=params['fancy_damping'], do_linesearch=params["pol_linesearch"], min_lm = params["pol_cg_min_lm"])
    elif params['pol_opt_alg']=="cg_fixed_lm": # FOR ICML
        g,s = Glf.f_gradtrainloss(th_before, th_before)
        from control4.optim.krylov import cg
        pol_cg_damping = params["pol_cg_damping"]
        dth = cg(lambda p: Glf.f_trainfvp(th_before,p,th_before) + p*(pol_cg_damping*s), -g/params["cg_fixed_lm_value"], 
            verbose=True,cg_iters=params['pol_cg_iters'])
        th_after = th_before + dth
    elif params['pol_opt_alg']=='empirical_fim': # FOR ICML
        gradlogp_tz = Glf.f_gradlogp()        
        A_zz = np.cov(gradlogp_tz.T)
        th_after = cg_optimize(th_before, timers.wrap(lambda th: Glf.f_trainloss(th, th_before),"trainloss"), 
                    timers.wrap(lambda th: Glf.f_gradtrainloss(th, th_before),"gradtrainloss"),
                    params['pol_cg_kl'], params['pol_cg_steps'], params['pol_cg_damping'],
                    cg_iters=params['pol_cg_iters'], fmetric=timers.wrap(lambda th,p: A_zz.dot(p),"fvp"), 
                    fancy_damping=params['fancy_damping'], do_linesearch=params["pol_linesearch"], min_lm = params["pol_cg_min_lm"])
    elif params['pol_opt_alg']=='maxkl':
        assert not params['fancy_damping']
        with Message("computing FIMs"):
            As = Glf.f_hessians(th_before)
        Q = As.shape[1]
        As[:,np.arange(Q),np.arange(Q)] += 1e-4
        from control4.optim.lomqc import lomqc
        g = Glf.f_gradtrainloss(th_before, th_before)
        with Message("optimizing"):
            step = lomqc(g, As, params['pol_cg_kl'])
        scaling = 1
        while True:
            th_after = th_before+step*scaling
            allkl = Glf.f_allkl(th_after)
            maxkl = allkl.max()
            newloss =  Glf.f_trainloss(th_after, th_before)
            print "scaling",scaling,"maxkl:",maxkl,"improvement",train_losses_before[-1]-newloss
            if maxkl < params['pol_cg_kl']*2 and newloss < train_losses_before[-1]: 
                print "ls success"
                break
            else:
                scaling *= .5
    else:
        raise NotImplementedError

    train_losses_after = Glf.f_losses(th_after,"train",th_before)
    test_losses_after = 0*train_losses_after if params['disable_test'] else Glf.f_losses(th_after,"test",th_before)

    timers.disp("Optimization timings")

    rows = []
    delta_train_losses = train_losses_before - train_losses_after
    delta_test_losses = test_losses_before - test_losses_after

    headers = ["cost","train before","dtrain","dtest","ratio"]
    for (name,train_loss, delta_train_loss,delta_test_loss) in zip(G.pol_loss_names, train_losses_before, delta_train_losses, delta_test_losses):
        rows.append((name,train_loss, delta_train_loss, delta_test_loss,delta_test_loss/delta_train_loss))
    print "=========== Policy optimization results (%s) ==========="%params['pol_opt_alg']
    print tabulate(rows,headers=headers)


    ecidx = G.pol_loss_names.index("ec")
    klidx = G.pol_loss_names.index("kl")

    diags["kloldnewtrain"].append(-delta_train_losses[klidx])
    diags["kloldnewtest"].append(-delta_test_losses[klidx])
    diags["delta_ec_train"].append(delta_train_losses[ecidx])
    diags["delta_ec_test"].append(delta_test_losses[ecidx])
    diags["normstepsize"].append(np.linalg.norm(th_after - th_before))

    # if params['cdp'] == "mean_std": print "stdev:", np.exp(policy.cdp.b2s_e.get_value())
    # if params['pa'] == "nn+javg": print "averaging coeffs", policy.pol_approx.j_moving_avg_coeffs.get_value()

    ######
    G.agent.policy.set_var_values_flat(th_after)


def optimize_vf(params, agent, diags): #pylint: disable=W0613


    log = logging.getLogger("optimize_policy")
    log.info("Optimizing value function")

    th_before = agent.vf.var_values_flat()

    timers = Timers()

    train_losses_before = Glf.f_vflosses(th_before,"train",th_before)
    test_losses_before = 0*train_losses_before if params['disable_test'] else Glf.f_vflosses(th_before,"test",th_before)

    if params['vf_opt_alg']=="cg":
        th_after = cg_optimize(th_before, timers.wrap(lambda th: Glf.f_vftrainloss(th, th_before),"trainloss"), timers.wrap(lambda th: Glf.f_vfgradtrainloss(th, th_before),"gradtrainloss"),
            params['vf_cg_kl'], params['vf_cg_steps'], params['vf_cg_damping'],
            cg_iters=params['vf_cg_iters'], fmetric=timers.wrap(lambda th,p: Glf.f_vftrainfvp(th,p,th_before),"fvp"),
            fancy_damping=params['fancy_damping'],min_lm=params["vf_cg_min_lm"])
    else:
        raise NotImplementedError

    train_losses_after = Glf.f_vflosses(th_after,"train",th_before)
    test_losses_after = 0*train_losses_after if params['disable_test'] else Glf.f_vflosses(th_after,"test",th_before)

    timers.disp("Optimization timings")

    rows = []
    delta_train_losses = train_losses_before - train_losses_after
    delta_test_losses = test_losses_before - test_losses_after

    headers = ["cost","train before","dtrain","dtest","ratio"]
    for (name,train_loss, delta_train_loss,delta_test_loss) in zip(G.vf_loss_names, train_losses_before, delta_train_losses, delta_test_losses):
        rows.append((name,train_loss, delta_train_loss, delta_test_loss,delta_test_loss/delta_train_loss))
    print "=========== VF optimization results (%s) ==========="%params['vf_opt_alg']
    print tabulate(rows,headers=headers)

    # if params['cdp'] == "mean_std": print "stdev:", np.exp(policy.cdp.b2s_e.get_value())
    # if params['pa'] == "nn+javg": print "averaging coeffs", policy.pol_approx.j_moving_avg_coeffs.get_value()

    ######
    G.agent.vf.set_var_values_flat(th_after)


def need_policy_opt(_params):
    return True

def need_vf_opt(params):
    return params['vf_opt_mode'] == "separate"

def is_slave(params):
    return bool(params.get('slave_addr'))



def do_snapshots(iteration, params, hdf_filename, agent, path_diags,opt_diags):
    hdf = h5py.File(hdf_filename, "r+")


    policy_iter = params['policy_iter']

    if is_save_iter(iteration, params['save_data_snapshots'], policy_iter):
        raise NotImplementedError
        # snapgrp = hdf["data_snapshots"].create_group(snapname(iteration))
        # namedtuplelist_to_hdf(paths, snapgrp.create_group("paths"))
    if is_save_iter(iteration, params['save_policy_snapshots'], policy_iter):
        save_agent_snapshot(hdf, agent, iteration)

    # Save diagnostics occcasionally
    if is_save_iter(iteration, 10, policy_iter):
        write_diagnostics(hdf, path_diags)
        write_diagnostics(hdf, opt_diags)

    hdf.close()

def initialize_agent(agent,params):
    initfrom_file = fetch_file(params['initialize_from'])
    loadhdf = h5py.File(initfrom_file,"r")
    assert len(loadhdf["agent_snapshots"]) > 0
    grpname = loadhdf["agent_snapshots"].keys()[params['initialize_from_idx']]
    print "loading snapshot",grpname
    agent.from_h5(loadhdf["agent_snapshots"][grpname])
    loadhdf.close()        
    agent.pprint()
    
def single_path_spi(params):

    # So master and slave can concurrently be compiling functions:
    if params['par_mode'] == "cluster" and not is_slave(params): G.pool = create_cluster_pool(params)

    if params["test_mode"]: configure_test_mode(params)
    if params['par_mode']=="off": params['n_processes']=1

    np.random.seed(params['seed'])
    seed_iter = itertools.count(params['seed']*100000)

    # MDP 
    mdp = get_mdp(params['mdp_name'], string2dict(params['mdp_kws']))

    # Policy
    agent = construct_agent(params,mdp)
    agent.pprint()


    if params['initialize_from']:
        initialize_agent(agent,params)

    # VF
    path_diags = defaultdict(list) # Diagnostics measured after collecting paths
    opt_diags = defaultdict(list) # Diagnostics measured after optimization step
    # Need to be separate them so we can print the last element of each timeseries

    # Output file
    if not is_slave(params): 
        _hdf = setup_outfile(params, agent)
        hdf_filename = _hdf.filename
        _hdf.close()

    # Loss funcs
    with Message("Compiling loss functions"):
        make_loss_funcs(params,mdp,agent)

    t_start = time.time()

    G.mdp = mdp
    G.params = params
    G.agent = agent
    

    if is_slave(params):
        # This is a slave instance
        print "going into slave loop %s"%params['slave_addr']
        from control4.cloud.slave_loop import slave_loop
        slave_loop(params['slave_addr'])    
    
    if params['par_mode'] != "cluster": G.pool = create_non_cluster_pool(params)


    for policy_iteration in xrange(params['policy_iter']):

        with Message("Policy iter %i"%policy_iteration):

            with Message("Do simulations"):
                generate_paths_toplevel(params, seed_iter)
            
            path_info = G.pool.gather(pathinfomapper,merge_dicts,None)
            compute_path_diagnostics(mdp, path_diags, path_info, params["diag_traj_length"])



            if params['adaptive_lambda']:
                raise NotImplementedError
            else:
                advs_vs = G.pool.gather(compute_and_store_advantages, concatenate, None)
                all_advs, all_vtargs, all_vcurs = zip(*advs_vs)
                if params['episodic_baseline']: baseline = compute_baseline(all_advs)
                else: baseline = None
                if params['standardize_adv']: scaling = np.concatenate(all_advs).std()
                else: scaling = 1.0
                G.pool.apply(shift_and_scale_advantages, (baseline, scaling))
                
                bigvtarg = np.concatenate(all_vtargs)
                vtargvar = bigvtarg.var()
                bigvcur = np.concatenate(all_vcurs)
                vcurvar = bigvcur.var()
                ev_prev = explained_variance_1d(bigvcur, bigvtarg)
                print "targ var: %f. cur var: %f. ev: %f"%(vtargvar, vcurvar, ev_prev)
                path_diags["vf_ev"].append(ev_prev)
                path_diags["vf_targ"].append(vtargvar)
                path_diags['vf_cur'].append(vcurvar)
                G.pool.apply(set_vvar, np.square(bigvtarg-bigvcur).mean().astype(floatX))

            if need_policy_opt(params):
                with Message("Optimize policy"):
                    optimize_policy(params, agent, opt_diags)

            if need_vf_opt(params):
                with Message("Optimize value function"):
                    if 1:
                        optimize_vf(params, agent, opt_diags)
                    else:
                        paths = G.pool.gather(get_paths_mapper, concatenate,None)
                        from control4.algs.fit_linear_vf import fit_linear_vf_single_path                    
                        fit_linear_vf_single_path(agent.vf, paths, params)
                        G.pool.apply(set_vf_var_values,agent.vf.var_values())


            with Message("Snapshots"):
                do_snapshots(policy_iteration, params, hdf_filename, agent, path_diags,opt_diags)



            opt_diags["times"].append(time.time() - t_start)


def configure_test_mode(params):
    """
    Change parameter setting so we can run this script really fast
    """    
    params['policy_iter'] = 2
    if "paths_per_batch" in params and params['paths_per_batch']>0: params['paths_per_batch'] = 5
    if "timesteps_per_batch" in params and params['timesteps_per_batch']>0: params['timesteps_per_batch'] = 2000
    params['outfile'] = "/tmp/test_mode_output%i.h5"%os.getpid()
    if osp.exists(params['outfile']): os.unlink(params['outfile'])
    params['path_length'] = 100
    theano.config.optimizer = "None"
    import atexit
    def cleanup():
        if osp.exists(params["outfile"]):
            os.unlink(params["outfile"])
    atexit.register(cleanup)
