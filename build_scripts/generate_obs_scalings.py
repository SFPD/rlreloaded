#!/usr/bin/env python
from glob import glob
import os.path as osp
import h5py
from control3.common_util import *


indir = osp.expandvars("$CTRL_DATA/misc/mdp_random_trajs")
outdir = osp.expandvars("$CTRL_DATA/misc/mdp_obs_ranges")
mkdirp(outdir)

for fname in glob(osp.join(indir,"*.h5")):
    print "using", fname
    basename = osp.basename(fname)[:-3]
    hdf = h5py.File(fname,'r')
    obs = np.concatenate([grp["os"].value for grp in hdf.values()])
    lo,hi = np.percentile(obs, [10,90], axis=0)
    outfile = osp.join(outdir,"%s.npy"%basename)
    np.save(outfile,[lo,hi])

