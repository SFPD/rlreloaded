import numpy as np

def compute_path_diagnostics(mdp, diags, path_info, t_short, disp=True):
    length_n = path_info.get("length")
    if length_n is not None:
        length_n = np.array(length_n)
        diags["length_mean"].append(length_n.mean())
        diags["length_stderr"].append(length_n.std() / np.sqrt(len(length_n)-1))
        n_timesteps_prev = diags.get("total_num_timesteps",[0])[-1]
        diags["total_num_timesteps"].append(n_timesteps_prev+length_n.sum())


    li_entropy_n = path_info.get("entropy")
    if li_entropy_n is not None:
        entropy_n = np.concatenate(li_entropy_n)
        assert entropy_n.ndim == 1
        total_length = entropy_n.size
        diags["entropy"].append( entropy_n.sum() / total_length) 

    # li_length, li_c_v, li_earlyc_v,li_entropy = zip(*path_summaries)

    li_c_tv = path_info.get("cost")
    if li_c_tv is not None:
        ctotalsum_v = np.zeros(li_c_tv[0].shape[1]) # sum over entire length of traj
        cearlysum_v = np.zeros(li_c_tv[0].shape[1]) # just look at first t_short timesteps, to calculate mean cost
        for cost_tv in li_c_tv:
            ctotalsum_v += cost_tv.sum(axis=0)
            cearlysum_v += cost_tv[:t_short].sum(axis=0)
        n_paths = len(li_c_tv)
        early_timesteps = t_short * n_paths

        avgcost_v = cearlysum_v / early_timesteps
        episodecost = ctotalsum_v.sum() / n_paths

        for (cost_name, val) in zip(mdp.cost_names(), avgcost_v):
            diags["avgcost_%s"%(cost_name)].append(val)
        diags["avgcost_total"].append( avgcost_v.sum() )

        diags["episodecost"].append(episodecost)

        for name, cost in mdp.unscaled_cost(dict(zip(mdp.cost_names(), avgcost_v))).iteritems():
            diags[name].append(cost)



    if disp:
        from tabulate import tabulate
        print tabulate([(name,ts[-1]) for (name,ts) in sorted(diags.items())])


def plot(mdp,agent):
    from control4.core.rollout import rollout
    def plot_callback(arrs): 
        mdp.plot(arrs)
    for i in xrange(plot):
        print "rollout %i/10"%(i+1)
        rollout(mdp,agent,999999,save_arrs=("u",),callback=plot_callback)            

    # import matplotlib.pyplot as plt
    # plt.clf()
    # plt.ioff()
    # for path in paths:
    #     plt.plot(path.x_td[:,0],path.x_td[:,1])
    # plt.ion()
    # plt.pause(.05)
    # plt.show()

def write_diagnostics(hdf,diags):
    diag_grp = hdf["diagnostics"]
    for (diag_name, val) in diags.items():
        if diag_name in diag_grp: 
            del diag_grp[diag_name]
        diag_grp[diag_name] = val

