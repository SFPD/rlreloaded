from glob import glob
from control4.config import CTRL_ROOT
import os.path as osp

def filelist_from_patterns(pats, rootdir=None):
    if rootdir is None: rootdir = CTRL_ROOT
    # filelist = []
    fileset = set([])
    lines = [line.strip() for line in pats]
    for line in lines:
        pat  = line[2:]
        newfiles = glob(osp.join(rootdir,pat))
        if line.startswith("+"):
            fileset.update(newfiles)
        elif line.startswith("-"):
            fileset.difference_update(newfiles)
        else:
            raise ValueError("line must start with + or -")
    filelist = list(fileset)
    return filelist
            