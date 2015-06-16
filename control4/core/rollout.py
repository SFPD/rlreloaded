def rollout(mdp, agent, max_length, save_arrs=(),callback=None):
    """
    Do single rollout from random initial states and save results to a list
    Returns
    --------
    init_arrs: mapping name -> row vector, specifying initial condition
    traj_arrs: mapping name -> list of row vectors, specifying state at t=0,1,...,T-1    
    """
    init_arrs = {}
    init_arrs.update(mdp.initialize_mdp_arrays())
    init_arrs.update(agent.initialize_lag_arrays())

    traj_arrs = {name:[] for name in save_arrs}

    cur_arrs = init_arrs.copy()
    for _ in xrange(max_length):
        cur_arrs.update(agent.call(cur_arrs))
        cur_arrs.update(mdp.call(cur_arrs))
        if callback is not None: callback(cur_arrs)
        for name in save_arrs: traj_arrs[name].append(cur_arrs[name])
        if cur_arrs.get("done",False):
            break
    return (init_arrs, traj_arrs)

