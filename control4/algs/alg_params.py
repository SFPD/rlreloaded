from control4.misc.console_utils import colorize
import argparse

def string2dict(s):
    """
    turn a string like "a=2,b=3.5" into a dict
    """
    # strval = re.compile(r"^\w+$")
    d = {}
    for stmt in s.split(","):
        if stmt:
            (key,valstr) = stmt.split("=")
            try:
                val = eval(valstr)
                d[key] = val
            except (NameError,SyntaxError):
                d[key] = valstr
    return d

def str2numlist(s,typ):
    """
    Convert string of comma-separated ints into list of ints or floats
    typ = int or float
    """
    s = s.strip()
    return map(typ,s.split(",")) if len(s)>0 else []

class AlgParams(object):
    """
    Define a group of parameters that are relevant to some algorithm.
    For example, 

    class Blah(AlgParams):
        required_param = int # Initialize required parameter with a type
        optional_param = 10  # Initialize optional parameter with a value
    """
    _group_condition = "True"



    @classmethod
    def add_to_parser(cls, parser, grp_name=None):
        if grp_name is None: grp_name = cls.__name__
        group = parser.add_argument_group(grp_name)

        # If it's not a CONDITIONAL object, make it one, with condition always True
        for (attr_name,val_or_cond) in cls.items():
            if not isinstance(val_or_cond, CONDITIONAL):
                setattr(cls, attr_name, CONDITIONAL(val_or_cond, "True"))

        for (attr_name,conditional) in cls.items():
            arg_name = "--" + attr_name
            val = conditional.val
            if isinstance(val, type):
                group.add_argument(arg_name, type = val, help=" ")
            else:
                group.add_argument(arg_name, type = type(val), default = val, help=" ")
    @classmethod
    def items(cls):
        return [(key,val) for (key,val) in cls.__dict__.items() if not key.startswith("_")]

def validate_and_filter_args(cls_list, ns):
    """
    For each CONDITIONAL item found,
        if conditional holds true, then argument must be set
        else, then argument shouldn't be changed (value should be None or default)
    """
    d = ns.__dict__.copy()
    
    n_errs = [0]
    def set_error(s):
        print colorize(s,'red')
        n_errs[0] += 1


    for cls in cls_list:
        group_cond_holds = eval(cls._group_condition, d) #pylint: disable=W0212
        for (attr_name,conditional) in cls.items():

            val = conditional.val
            cond_holds = eval(conditional.cond, d) and group_cond_holds

            if cond_holds:
                if d[attr_name] is None: # required param with no default value was not specified
                    set_error("missing conditionally required argument %s. (condition: \"%s\", group condition: \"%s\".)"%(attr_name, conditional.cond, cls._group_condition)) #pylint: disable=W0212
            else:
                # Specified value for param with no default value, or optional param with a default value
                if isinstance(val,type)  and d[attr_name] is not None or (not isinstance(val,type)) and d[attr_name] != val:
                    set_error("Specified value for conditional argument %s, but conditional doesn't hold (condition: \"%s\", group condition: \"%s\")"%(attr_name, conditional.cond, cls._group_condition)) #pylint: disable=W0212
                del ns.__dict__[attr_name]
                # print "removed",attr_name
    if n_errs[0] > 0:
        raise RuntimeError("%i argument errors"%n_errs[0])

def add_params_to_parser(name2param, parser):
    for (name,param) in name2param.items():
        param.add_to_parser(parser, grp_name=name)


class CONDITIONAL(object):
    def __init__(self, val, cond):
        self.val = val
        self.cond = cond


class GeneralScriptParams(AlgParams):
    seed = 0
    n_processes = -1
    plot = 0
    par_mode = "off"
    outfile = ""
    test_mode = 0
    write_hdf = 1
    dump_pkls = 0
    metadata = ""
    check_gradient=0
    misc_kws = ""

class ClusterParams(AlgParams):
    cloud_provider = "gce"
    cluster_name = str
    slave_addr = ""    
    _group_condition = "par_mode == 'cluster'"

class ProfileParams(AlgParams):
    line_profile_funcs = ""
    enable_cprofile = 0

class DiagnosticParams(AlgParams):
    diagnostics_every = 1
    save_policy_snapshots = 5
    save_data_snapshots = 0
    num_diag_samples=50
    diag_traj_length = 500

class MDPParams(AlgParams):
    mdp_name = str
    mdp_kws = ""


class SinglePathEstimationParams(AlgParams):
    episodic_baseline = 1
    paths_per_batch = 0
    timesteps_per_batch = 0
    path_chunk_size = CONDITIONAL(16, "timesteps_per_batch != 0")
    path_length = int
    _group_condition = "alg == 'single_path'"

class SPIParams(AlgParams):
    policy_iter = 10

    agent_module = str
    initialize_from = ""
    initialize_from_idx = CONDITIONAL(-1,"initialize_from != 0")
    policy_cfg = ""
    policy_kws = ""
    vf_cfg = ""
    alg = "single_path"

    # Advantage estimation
    standardize_adv = 1
    adaptive_lambda = 0
    lam = 1.0 
    gamma = .99
    vf_lam = 1.0

    fancy_damping = 1

    disable_test = 0

    deepmind_cheat_mode = 0

    # # PREPROC
    # pp = str

    # # STATE UPDATER 
    # msu = "none"
    # n_obs_lags = CONDITIONAL(int, "msu == 'lagged_obs'")
    # n_j_lags = CONDITIONAL(int, "msu == 'lagged_js'")
    # m_dim = CONDITIONAL(int, "msu == 'rnn'")
    # rnn_arch = CONDITIONAL("mln", "msu == 'rnn'")
    # rnn_hidden_layer_sizes = CONDITIONAL("","msu == 'rnn'")

    # # POLICY APPROXIMATOR
    # pa = str
    # hidden_layer_sizes = CONDITIONAL(str, "pa=='nn'")
    # nonlinearity = CONDITIONAL(str, "pa=='nn'")
    # nn_use_m = 0
    # nn_use_o = 1
    # nn_incoming_norm_init = 1.0
    # convnet_file = CONDITIONAL(str, "pa == 'cnn'")

    # # CDP
    # cdp = str
    # std_init = CONDITIONAL(0.5, "cdp == 'mean_std'")
    # u_res = CONDITIONAL(int, "cdp in ('grid','factored_grid')")
    # tile_scales = CONDITIONAL(3, "pp == 'tile'")

    # min_discrete_prob = 0.0

    # # Noise
    # control_filter = "none" #{none, linear_filter}
    # control_filter_period = CONDITIONAL(float, "control_filter != 'none'")


class PolicyOptParams(AlgParams):    

    # Loss function
    pol_loss = "expected_cost"    
    pol_squash_hi = CONDITIONAL(1.1,"'squash' in pol_loss")
    pol_l2_coeff = 1e-5
    pol_vferr_coeff = 0.01
    pol_ent_coeff = 0.0


    # optimization algorithm
    pol_opt_alg = "cg"
    pol_cg_kl = 0.01
    pol_cg_iters = 10
    pol_hess_subsample = 10
    pol_cg_steps = 1
    pol_cg_damping = 0.01
    pol_linesearch = 1
    pol_cg_min_lm = 0.01

class VFOptParams(AlgParams):

    vf_opt_mode = "off" # {none, joint, separate,linear}
    vf_opt_alg = "cg"


    # Value function loss function
    vf_loss_type = "l2" #{l2,l1,quantile,l2_asym}
    vf_loss_quantile = 0.0
    vf_l2_coeff = 1e-4

    # CG for value function
    vf_cg_kl = .25
    vf_cg_iters = 10
    vf_hess_subsample = 10
    vf_cg_damping = 0.01
    vf_cg_steps = 1
    vf_cg_min_lm = 0.01

    vf_end_mode = "zero"

    # Linear value function
    lvf_fit_method = "lspe" # {lspe,lstd}
    lvf_regress_method = "ridge" # {ridge,lars}
    lvf_lars_alphas = CONDITIONAL(5,"lvf_regress_method=='lars'")
    lvf_obs_slice = "" # Only use this slice of coordinates
    lvf_subsample = 0 # Number of samples to use for fitting linear value function



def make_argument_parser():
    parser = argparse.ArgumentParser(formatter_class=lambda prog : argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=50))
    return parser    