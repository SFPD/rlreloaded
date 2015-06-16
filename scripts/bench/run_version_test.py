#!/usr/bin/env python
import os,sys
from control4.misc.console_utils import call_and_print
from control4.algs.save_load_utils import get_git_commit

from control4.config import CTRL_ROOT

os.chdir(CTRL_ROOT)
commit = get_git_commit(CTRL_ROOT)
if "--gce" in sys.argv: 
    commit = commit.split("-")[0]
    sys.argv.append("--one_run_per_machine")
call_and_print("$CTRL_ROOT/scripts/bench/run_experiment.py $CTRL_ROOT/maintenance/benchmarks.yaml \
    benchmarks4/%s --alter_default_settings=cfg_name=%s --n_runs=5  "%(commit,commit) + " ".join(sys.argv[1:]))