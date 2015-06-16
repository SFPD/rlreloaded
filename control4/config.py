import os.path as osp, json
CTRL_ROOT = osp.abspath(osp.dirname(osp.dirname(__file__)))
MJC_DATA_DIR = osp.join(CTRL_ROOT,"cpp/3rdparty/mjc/mjcdata")
import theano
floatX = theano.config.floatX #pylint: disable=E1101
from control4.misc.func_utils import memoized
from control4.misc.deep_dict import deep_in, deep_setitem
import os,socket

def setup_logging():
    import logging.config,yaml    
    with open(osp.join(CTRL_ROOT, "config/logging.yaml")) as fh:
        d = yaml.load(fh)
    logging.config.dictConfig(d)

def print_theano_config():
    from tabulate import tabulate
    fields = ["optimizer","compute_test_value","floatX","linker"]
    print tabulate([(field, getattr(theano.config,field)) for field in fields]  )
    print "hostname",socket.gethostname()    

def resolve_cfg_loc(s):
    return osp.join(CTRL_ROOT,s)

def user_cfg_dir():
    return osp.expanduser("~/.rlreloaded")

@memoized
def load_config():
    rcfileloc = osp.join(user_cfg_dir(),"rlreloadedrc")
    fallbackloc = osp.join(CTRL_ROOT, "config/rlreloaded.json")
    if osp.exists(rcfileloc):
        useloc = rcfileloc
    if not osp.exists(rcfileloc):
        print "warning: couldn't find rcfile at %s, falling back to default at %s"%(rcfileloc, fallbackloc)
        useloc = fallbackloc
    with open(useloc,"r") as fh:
        d = json.load(fh)
    env_update_config(d)
    return d

def env_update_config(d):
    for (key,val) in os.environ.items():
        if key.startswith("CTRL__"):
            if not deep_in(d, key.split("__")[1:]):
                raise RuntimeError("environment variable %s doesn't correspond to a config option"%key)
            deep_setitem(d, key.split("__")[1:], val)
    return d
