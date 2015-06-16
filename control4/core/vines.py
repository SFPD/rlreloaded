import numpy as np
from collections import namedtuple
from control4.misc.randomness import random_indices
from control4.config import floatX
import theano,theano.tensor as TT #pylint: disable=F0401
from control4.maths.discount import discountedsum

BranchData = namedtuple("BranchData", ["cxy_nk", "cyz_nkst", "vx_n", "vyz_nkst", "length_nks", "b_nkb", "sp_nk", "sq_nk"])
RolloutSetData = namedtuple("RolloutSetData",["x_nd","o_no","a_na"])


def sample_from_lists(list_lengths, n_samples):
    """
    Inputs:
        :list_lengths: lengths of lists
        :n_samples: total number of samples wanted
    Returns:
        list of list of indices: indices from first list, indices from second list, and so on
    """
    n_total = sum(list_lengths)
    flat2whichli = np.zeros(n_total,'int')
    flat2idxinli = np.zeros(n_total,'int')
    pos = 0
    for (i_li, ll) in enumerate(list_lengths):
        flat2whichli[pos:pos+ll] = i_li
        flat2idxinli[pos:pos+ll] = np.arange(ll)
        pos += ll

    out = [ [] for _ in list_lengths]

    flat_inds = random_indices(n_total, n_samples)
    for flat_ind in flat_inds:
        out[ flat2whichli[flat_ind] ].append( flat2idxinli[flat_ind] )
    return out


def sample_rollout_set(inittrajs,rollout_set_size):

    indlists = [np.array(li) for li in sample_from_lists( [len(traj["a"]) for (init,traj) in inittrajs] , rollout_set_size)]

    x_nd = np.concatenate([traj["x"][i-1] if i>0 else init["x"] for ((init,traj),indlist) in zip(inittrajs,indlists) for i in indlist])
    o_no = np.concatenate([traj["o"][i-1] if i>0 else init["o"] for ((init,traj),indlist) in zip(inittrajs,indlists) for i in indlist])
    a_na = np.concatenate([traj["a"][i] for ((init,traj),indlist) in zip(inittrajs,indlists) for i in indlist])

    rsd = RolloutSetData(x_nd,o_no,a_na)
    return rsd


# XXX do_rollouts_frozen doesn't work if there are terminal states
def do_rollouts_frozen_noise(mdp, call_frozen, rand_tnr, init_arrs,save_arrs=()):
    traj_arrs = {name:[] for name in save_arrs}
    cur_arrs = init_arrs.copy()
    for rand_nr in rand_tnr:
        cur_arrs.update(call_frozen(cur_arrs, rand_nr))
        cur_arrs.update(mdp.call(cur_arrs))
        for name in save_arrs: traj_arrs[name].append(cur_arrs[name])
    return traj_arrs


class FrozenNoisePolicy(object):
    def __init__(self,agent):
        r = TT.matrix("r",floatX)
        self.agent = agent
        input_dict = agent.symbolic_inputs()
        input_list = agent._input_list(input_dict) #pylint: disable=W0212
        input_list.append(r)
        output_dict = self._symbolic_call(input_dict, r)
        output_list = [output_dict[name] for name in agent._output_names] #pylint: disable=W0212
        self._call = theano.function(input_list, output_list, on_unused_input='ignore')

    def _symbolic_call(self, input_dict, r):
        """
        call with symbolic arrays
        """
        cpd = self.agent.cpd()
        output = self.agent.ponder(input_dict)
        b = cpd.draw_frozen(output["a"],r)
        output.update({
            "b":b,
            "q":cpd.liks(output["a"],b),
            "u":self.agent.b2u(b)
        })
        return output

    def __call__(self,input_dict, r):
        outputs = self._call(*(self.agent._input_list(input_dict)+[r])) #pylint: disable=W0212
        return zip(self.agent._output_names, outputs) #pylint: disable=W0212

# HACK:
def get_liks_numeric(agent): 
    
    if not hasattr(agent, "liks_numeric"):
        a = TT.matrix("a",agent.output_dtype("a"))
        b = TT.matrix("b",agent.output_dtype("b"))
        p = agent.cpd().liks(a,b)
        agent.liks_numeric = theano.function([a,b],p)
    return agent.liks_numeric

def generate_branches(mdp,agent,rsd, branch_length, actions_per_state, trials_per_action, all_actions, fs_b2u):
    rollout_set_size = rsd.x_nd.shape[0]

    cpd = agent.cpd()

    T = branch_length
    K = actions_per_state
    S = trials_per_action
    N = rollout_set_size
    B = cpd.b_dim()
    R = cpd.r_dim()

    x_nk_d = np.repeat( rsd.x_nd, K, axis=0) # x_0

    if all_actions:
        # b_nkb = cpd.get_all_of_each(rsd.a_na) # j_0, p(j_0)
        raise NotImplementedError
    else:
        b_nkb = cpd.draw_multi(rsd.a_na, K)
        sq_nk = get_liks_numeric(agent)(np.repeat(rsd.a_na,K,axis=0), b_nkb.reshape(N*K,B)).reshape(N,K)
        # for sanity checking, see if we end up with same rollout when we start with the same action
        # b_nkb[:,1,:] = b_nkb[:,0,:]

    b_nk_b = b_nkb.reshape(N*K,-1)
    u_nk_e = fs_b2u(b_nk_b)

    next_arrs = mdp.call({"x":x_nk_d,"u":u_nk_e})
    nextx_nk_d = next_arrs["x"]
    cxy_nk = next_arrs["c"].sum(axis=-1)
    nexto_nk_o = next_arrs["o"]

    nextx_nks_d = np.repeat(nextx_nk_d,S,axis=0)
    nexto_nks_o = np.repeat(nexto_nk_o,S,axis=0)
    b_nks_b = np.repeat(b_nk_b,S,axis=0)

    init_arrs = {"x":nextx_nks_d,"o":nexto_nks_o,"b":b_nks_b}

    vf_enabled = "v" in agent.output_info()

    save_arrs = ["c"]
    if vf_enabled: 
        save_arrs.append("v")
        vx_n = next_arrs["v"]
    else:
        vx_n = np.zeros(N,floatX)

    ### TODO: RECOMPILES EVERY TIME!
    call_frozen = FrozenNoisePolicy(agent)
    rand_tnsr = np.random.rand(T, N, S, R).astype(floatX)
    rand_t_nks_r = np.repeat(rand_tnsr, K, axis=1).reshape(T,N*K*S,R)
    # No CRN:
    # rand_t_nks_r = np.random.rand(T,N*K*S,R).astype(floatX)
    traj_arrs = do_rollouts_frozen_noise(mdp, call_frozen, rand_t_nks_r, init_arrs, save_arrs)
    cyz_qt = np.array(traj_arrs["c"]).sum(axis=-1).T
    if vf_enabled:
        vz_nt = np.concatenate(traj_arrs["v"],axis=0) if "v" in traj_arrs else np.zeros((N,K,S,T+1),floatX)
    else:
        vz_nt = np.zeros((N*K*S,T+1),floatX)
    length_n = np.ones(N*K*S,'int64')*N # XXX generate_branches doesn't work for variable length

    bd = BranchData(cxy_nk.reshape(N, K), cyz_qt.reshape(N,K,S,T), vx_n, vz_nt.reshape(N,K,S,T+1), length_n.reshape(N,K,S), 
        b_nkb.reshape(N,K,B), sq_nk,sq_nk)

    return bd
    

def compute_branch_advantages(bd,gamma,lam):
    _,K = bd.cxy_nk.shape
    vx_nk1 = np.repeat(bd.vx_n[:,None,None],K,axis=1)
    vyz_nkst = bd.vyz_nkst.copy()
    nanmask_nkst = np.isnan(vyz_nkst)
    if nanmask_nkst.any():
        assert lam==1.0
        vyz_nkst[nanmask_nkst] = 0
        # for lam=1 only the last value matters
        # if the trajectory (n,a,s) didn't end, then vyz_nkst[n,a,s] has finite value at the end and nans before it. we set those to zero but it doesn't matter
        # if the trajectory (n,a,s) ended, then setting final v to zero is correct
    vyz_nkt = vyz_nkst.mean(axis=2)
    cxy_nk1 = bd.cxy_nk[:,:,None]
    cyz_nkt = bd.cyz_nkst.mean(axis=2)
    v_nkt = np.concatenate([vx_nk1, vyz_nkt],axis=2)
    c_nkt = np.concatenate([cxy_nk1, cyz_nkt],axis=2)
    delta_nkt = c_nkt + gamma*v_nkt[:,:,1:] - v_nkt[:,:,:-1]
    adv_nk = discountedsum(delta_nkt, gamma*lam,axis=2)
    return adv_nk.astype(floatX)


