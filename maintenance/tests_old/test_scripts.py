from control3.common_util import *
import control3
import os.path as osp
os.chdir(osp.join(control3.CTRL_ROOT, "scripts"))
scripts = [
]
for cmd in scripts:
    call_and_print(cmd)