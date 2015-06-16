#!/usr/bin/env python

import argparse, subprocess
parser = argparse.ArgumentParser()
parser.add_argument("--dry",action="store_true")
parser.add_argument("--patfile", type=argparse.FileType("r"))
parser.add_argument("--run",type=int,default=1)
args = parser.parse_args()

from glob import glob
import os, os.path as osp
import imp
from control4.misc.console_utils import call_and_print, colorize
from control4.config import CTRL_ROOT

def have_module(modname):
    try:
        imp.find_module(modname)
        return True
    except ImportError:
        return False

def get_default_cmds():

    ignore = ["setup.py"]
    # if not ("DISPLAY" in os.environ): ignore.extend(["test_scripts.py"])
    if not have_module("pygame"): ignore.extend(["pygameviewer.py"])
    
    pats = ["control4/*.py","control4/*/*.py","maintenance/tests/*.py"]
    pyfiles = []
    for pat in pats: pyfiles.extend(glob(osp.expandvars("$CTRL_ROOT/%s"%pat)))


    cmds = []

    for pyfile in pyfiles:
        if osp.basename(pyfile) not in ignore:
            cmds.append("python %s"%pyfile)

    cmds.extend([
        "python $CTRL_ROOT/scripts/diagnostics/sim_policy.py  --agent_module=control4.agents.random_atari_agent --mdp_name=atari:breakout --one_traj --max_steps=100",
        "python $CTRL_ROOT/scripts/diagnostics/sim_policy.py --agent_module=control4.agents.random_continuous_agent --mdp_name=mjc:3swimmer --one_traj --max_steps=100",
        "python $CTRL_ROOT/scripts/bench/run_experiment.py $CTRL_ROOT/maintenance/otheralgs.yaml --pipe=off /tmp/blah --test",
        "python $CTRL_ROOT/scripts/bench/run_experiment.py $CTRL_ROOT/maintenance/benchmarks.yaml --pipe=off /tmp/blah --test",
    ])

    return cmds


def main():
    os.chdir(CTRL_ROOT)
    os.environ["THEANO_FLAGS"] = "floatX=float64,optimizer=None"

    if args.patfile is None:
        cmds = get_default_cmds()
        patfile = "/tmp/test_cmds.txt"
        print "saving list of commands to %s"%patfile
        with open(patfile,"w") as fh:
            for cmd in cmds:
                fh.write(cmd + "\n")
            
    else:
        cmds = filter(lambda x:len(x.strip())>0,args.patfile.readlines())


    if args.run:
        fails = []
        for cmd in cmds:
            if args.dry:
                print colorize(cmd,'green')
            else:
                try:
                    call_and_print(cmd, check=True)
                except subprocess.CalledProcessError:
                    print colorize("TEST FAILED", "red", bold=True)
                    fails.append(cmd)

        print "======== SUMMARY ======="
        n_total = len(cmds)
        n_fail = len(fails)
        n_succ = n_total - n_fail
        print "Ran %i scripts. %i succeeded. %i failed."%(n_total, n_succ, n_fail)
        if n_fail > 0:
            print "Failures:"
            failfile = "/tmp/test_fails.txt"
            with open(failfile,"w") as fh:
                for fail in fails:
                    print fail
                    fh.write(fail + "\n")
            print "Failures written to %s"%failfile


if __name__ == "__main__":
    main()