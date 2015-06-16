#!/usr/bin/env python


import sys
from control4.misc.console_utils import call_and_print
cmd = "python $CTRL_ROOT/scripts/bench/analyze_experiment.py $CTRL_DATA/results/benchmarks4 --stat_names=avgcost_total --versions --avg_runs --table_style=1 --label_lines_by=cfg " + " ".join(sys.argv[1:])
call_and_print(cmd)


