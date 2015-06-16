import numpy as np
from control4.config import floatX

def gen_scaling_data(mdp):
    # TODO: Try different strategies for getting the scaling

    n_total_min = 10000
    max_traj_len = 200

    from control4.core.rollout import rollout
    from control4.agents.random_continuous_agent import RandomContinuousAgent

    obs_arrs = []
    total_timesteps = 0

    agent = RandomContinuousAgent(mdp)

    while total_timesteps < n_total_min:
        _init_arrs,traj_arrs = rollout(mdp,agent,max_traj_len,save_arrs=("o",))
        obs_arr = np.concatenate(traj_arrs['o'])
        obs_arrs.append(obs_arr)
        total_timesteps += len(obs_arr)

    obs = np.concatenate(obs_arrs,axis=0)
    result = np.array(np.percentile(obs, [10,90], axis=0)).astype(floatX)

    return result[0], result[1]

def main():
    from control4.algs.save_load_utils import get_mdp
    mdp = get_mdp("mjc:3swimmer")
    gen_scaling_data(mdp)
    mdp = get_mdp("mjc2:3d_humanoid")
    gen_scaling_data(mdp)

if __name__ == "__main__":
    main()