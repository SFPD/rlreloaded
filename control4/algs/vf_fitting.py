from control4.maths.discount import discount
from control4.maths.numeric import explained_variance_1d
from control4.misc.randomness import random_indices
from control4.misc.console_utils import Message
from control4.config import floatX
import numpy as np



def fit_linear_vf_single_path(vf, paths, params):
    if params["lvf_obs_slice"]:
        ind_from,ind_to = map(int,params["lvf_obs_slice"].split(","))
        assert vf.nn_use_o and not vf.nn_use_m
        good_inds = np.arange(ind_from,ind_to)
        mean_mo_Q = np.zeros(len(good_inds),floatX)
        std_mo_Q = np.ones(len(good_inds),floatX)
    else:
        mo_nq = np.concatenate([vf.select_mo(path.prevm_tg[1:path.T+1], path.o_tf[:path.T]) for path in paths],axis=0)
        mean_mo_q = mo_nq.mean(axis=0).astype(floatX)
        std_mo_q  = mo_nq.std(axis=0).astype(floatX)
        tiny = 1e-8
        good_inds = np.flatnonzero(std_mo_q > tiny)
        mean_mo_Q = mean_mo_q[good_inds]
        std_mo_Q = std_mo_q[good_inds]

    # Set the values in the shared variables
    vf.good_inds.set_value(good_inds)
    vf.mean_mo_Q.set_value(mean_mo_Q)
    vf.std_mo_Q.set_value(std_mo_Q)

    # Standardized features for each trajectory
    train_phis = [vf.fs_feats(path.prevm_tg[1:path.T+1], path.o_tf[:path.T]) for path in paths]
    train_cs = [path.c_tv[:path.T].sum(axis=1) for path in paths]
    train_vs = [path.v_t for path in paths]

    gamma = params["gamma"]
    lam = params["vf_lam"]

    # Construct regressors
    if params["lvf_fit_method"] == "lspe": 
        X = np.concatenate(train_phis)
        # yold = np.concatenate([discount(np.concatenate([c,np.array([v[-1]])])[:-1],gamma) for (c,v) in zip(train_cs,train_vs)])
        deltas = [ c + gamma*v[1:] - v[:-1] for (c,v) in zip(train_cs,train_vs) ]
        y = np.concatenate([v[:-1] + discount(delta, lam*gamma) for (v,delta) in zip(train_vs,deltas)])

        w_Q,b,_ = linear_regression(X,y,params)
        # Coefficients

    elif params["lvf_fit_method"] == "lstd":
        nfeats = train_phis[0].shape[1]+1
        b = np.zeros(nfeats)
        A = np.zeros((nfeats,nfeats))
        for (c,phi) in zip(train_cs,train_phis):
            phi1 = np.concatenate([phi, np.ones((len(phi),1))],axis=1)
            b += discount(c,gamma*lam).dot(phi1)
            filtphi = discount(phi1, gamma*lam)
            filtphi = filtphi[:-1] - gamma*filtphi[1:]
            A += phi1[:-1].T.dot( filtphi )

        # XXX needs constant term
        w = np.linalg.solve(A,b)
        w_Q = w[:-1]
        b = w[-1]
                
    else:
        raise NotImplementedError

    vf.w_Q.set_value(w_Q.astype(floatX))
    vf.b.set_value(np.array(b,floatX))


def linear_regression(X,y,params):
    import warnings
    from sklearn import linear_model


    if params["lvf_subsample"]:
        subsample_inds = random_indices(len(X),params["lvf_subsample"])
        X = X[subsample_inds]
        y = y[subsample_inds]

    if params["lvf_regress_method"] == "ridge":
        alphas =  [.01, .1, 1., 10.]
        model = linear_model.RidgeCV(normalize=True,store_cv_values=True,alphas =alphas) #pylint: disable=E1120,E1123
        model.fit(X,y)
        print "alpha idx: %i/%i"%(model.cv_values_.mean(axis=0).argmin(), len(alphas))
    elif params["lvf_regress_method"] == "lars":
        model = linear_model.LassoLarsCV(max_n_alphas=params["lvf_lars_alphas"], n_jobs=params["n_processes"], cv=3, max_iter=10) #pylint: disable=E1120,E1123
        with Message("fit lasso lars cv"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.astype('float64'), y.astype('float64'))
        print "LARS obtained %i nonzero coefficients"%(model.coef_ != 0).sum()
    else:
        raise NotImplementedError

    ev = explained_variance_1d(model.predict(X), y)
    print "explained variance", ev
    return model.coef_, model.intercept_,dict(ev=ev)