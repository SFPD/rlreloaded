#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--files",nargs="+")
parser.add_argument("--patfile", type=argparse.FileType("r"))
args = parser.parse_args()

import subprocess,os,os.path as osp
from control4.config import CTRL_ROOT
from control4.misc.test_utils import filelist_from_patterns
from control4.misc.console_utils import call_and_print,colorize


if args.files is None and args.patfile is None: args.patfile=open(osp.join(CTRL_ROOT,"maintenance/lintfiles.txt"),"r")


assert args.files is not None or args.patfile is not None
if args.files is not None:
    filelist = args.files
elif args.patfile is not None:
    filelist = filelist_from_patterns(args.patfile.readlines())
else:
    raise Exception("unreachable")

rcfile = osp.join(CTRL_ROOT,"maintenance/pylintrc")
# lint = "python /Library/Python/2.7/site-packages/pylint/lint.py"
lint = "pylint"
if filelist is not None:
    for fname in filelist:
        result = call_and_print("%s -f colorized --rcfile %s -r n %s"%(lint, rcfile, fname),check=False)
else:
    result = call_and_print("%s -f colorized  --rcfile %s -r n  *.py"%(lint,rcfile),check=False)

