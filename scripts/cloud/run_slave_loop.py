#!/usr/bin/env python
import sys
from control4.cloud.slave_loop import slave_loop
print sys.argv[1]
slave_loop(sys.argv[1])