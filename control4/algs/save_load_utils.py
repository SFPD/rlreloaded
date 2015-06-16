import importlib,datetime,h5py,os,os.path as osp,sys,subprocess
import theano
from control4.config import CTRL_ROOT,load_config
from control4.misc.console_utils import InDir,yes_or_no,call_and_print
from control4.misc.h5_utils import pkldump
from control4.algs.alg_params import string2dict


################################


def construct_agent(params,mdp):
    mod = importlib.import_module(params['agent_module'])
    construct = getattr(mod,"construct")
    return construct(params,mdp)

def get_mdp(name, kws=None):

    if kws is None: kws = {}
    if name.startswith("atari:"):
        game = name[len("atari:"):]
        from control4.mdps import atari
        return atari.AtariMDP(game,**kws)
    elif name.startswith("mjc:"):
        from control4.mdps import mjc
        mjcbasename = name.split(":")[1]
        return mjc.get_mjc_mdp_class(mjcbasename,kws)
    elif name.startswith("mjc2:"):
        from control4.mdps import mjc2
        mjcbasename = name.split(":")[1]
        return mjc2.MJCMDP(mjcbasename,**kws)        
    elif name == "pendulum":
        from control4.mdps import pendulum
        return pendulum.Pendulum(**kws)
    elif name == "cartpole_barto":
        from control4.mdps import cartpole
        return cartpole.CartpoleBarto(**kws)
    elif name == "nav2d":
        from control4.mdps import nav2d
        return nav2d.Nav2D(**kws)
    elif name == "quake":
        from control4.mdps import quake
        return quake.QuakeMDP(**kws)
    elif name == "textnav":
        from control4.mdps import textnav
        return textnav.TextNav(**kws)
    else:
        raise IOError("mdp %s not found"%name)

def load_agent_and_mdp(fname,snapshot_index):
    hdf = h5py.File(fname,"r")
    mdp = get_mdp(hdf["params/mdp_name"].value, string2dict(hdf["params/mdp_kws"].value))
    params_grp = hdf["params"]
    params = {}
    for (key,val) in params_grp.items():
        params[key]=val.value
    agent = construct_agent(params, mdp)
    
    snapgrpnames = sorted(hdf["agent_snapshots"].keys())
    print "found snapshots %s"%snapgrpnames
    print "loading",hdf["agent_snapshots"].values()[snapshot_index]
    agent.from_h5(hdf["agent_snapshots"].values()[snapshot_index])
    return agent, mdp, hdf

def gen_output_h5_name():
    """
    Generate filename based on date and time
    """
    return "/tmp/%s"%datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S.h5')
    
def snapname(i):
    return "%.4i"%i

    
def create_h5(fname):
    if osp.exists(fname):
        if yes_or_no("%s already exists. delete?"%fname):
            os.unlink(fname)
        else:
            sys.exit(1)
    hdf = h5py.File(osp.expandvars(fname))
    hdf.create_group("pkls")
    hdf.create_group("diagnostics")
    runinfo = hdf.create_group("runinfo")
    runinfo["commit"] = get_git_commit(CTRL_ROOT)
    runinfo["env"]=subprocess.check_output("env",shell=True)
    runinfo["theanoversion"]=theano.__version__
    runinfo["cmd"] = " ".join(sys.argv)
    return hdf


def get_git_commit(sourcedir):
    try:
        # http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
        with InDir(sourcedir):

            status = subprocess.check_output(["git","status"])
            clean = "working directory clean" in status

            label = subprocess.check_output(["git", "rev-parse","--short","HEAD"]).strip()
        if not clean:
            label += "-dirty"
    except subprocess.CalledProcessError:
        label = "unknown"
    return label

def dump_dict_to_hdf(hdf, path, d):
    grp = hdf.create_group(path) if path not in hdf else hdf["path"]
    for (key, val) in d.items():
        try:
            grp[key] = val
        except (ValueError,TypeError):
            print "dump dict: skipping",key

def setup_outfile(params,agent):
    outfilename = params['outfile'] if bool(params['outfile']) else gen_output_h5_name()
    print "Saving results to %s"%outfilename
    if params.get("write_hdf",True):
        hdf = create_h5(outfilename)
        dump_dict_to_hdf(hdf, "params", params)
        if params.get('dump_pkls'): pkldump(hdf, "/pkls/agent", agent)
        # if params.vf_external: pkldump(hdf, "/pkls/vf", vf)
    return hdf

def save_agent_snapshot(hdf,agent,iteration):
    grp = hdf.create_group("agent_snapshots/%s"%snapname(iteration))
    agent.to_h5(grp)

def is_save_iter(iteration, save_every, total):
    """
    {0, save_every, 2*save_every, ..., k*save_every, total-1}
    """
    return save_every and (iteration == (total-1) or iteration % save_every == 0)    

def fetch_file(name):
    """
    Pass in a filename or a uri like gs://adp4control/path/to/blah.h5
    If gs:// uri is provided, this function checks if the base filename is in $CTRL_DATA/shared.
    If not, fetch it and save it there.
    """

    config = load_config()
    if name.startswith("gs://"):
        bucket_path = config['cloud']['gce']['data_bucket']
        assert not bucket_path.endswith("/")
        if name.startswith(bucket_path):
            path = name[len(bucket_path)+1:]
        else:
            path = osp.basename(name)
            # raise NotImplementedError("currently I assume $CTRL_DATA is associated with your data bucket, so I don't know what to do with data in another bucket")
        local_fname = osp.join(os.getenv("CTRL_DATA"),path)
        if osp.exists(local_fname):
            print "already have file %s. not pulling from gs"%local_fname
        else:
            if not osp.exists(osp.dirname(local_fname)):
                os.makedirs(osp.dirname(local_fname))
            call_and_print("gsutil cp %(name)s %(local_fname)s"%dict(name=name,local_fname=local_fname))
        return local_fname
    else:
        assert osp.exists(name)
        return name
################################


