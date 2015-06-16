import h5py, yaml, re, cPickle, shutil
from control4.misc.console_utils import mkdirp,yes_or_no
from collections import OrderedDict
import os.path as osp


class ScriptRun(object):
    def __init__(self, info, script_name, run_idx, out_root):
        info = info.copy()
        self.info = info
        self.script_name = script_name
        self.run_idx = run_idx
        self.out_root = out_root

        assert osp.expandvars(out_root).startswith("/")

        self.test_name = self._expand_vars(info.pop("test_name","unnamed"))
        self.cfg_name = self._expand_vars(info.pop("cfg_name","unnamed"))
        self.command = info.pop("command").strip()
        self.add_extra_args = info.pop("add_extra_args",1)
        outfile_basename = info.pop("outfile",self.script_name+"_RUN%.2i.h5"%(self.run_idx))
        self.outfile =osp.join(self.out_root,outfile_basename)

    def _expand_vars(self, s):
        if s.startswith("$"):
            if s=="$script_name": 
                return self.script_name
            else:
                raise RuntimeError("unrecognized variable %s"%s)
        else:
            return s

    def get_cmd(self, pipe_to_logfile="off"):
        li = [self.command]
        li.extend(["--%s=%s"%(par,val) for (par,val) in self.info.items()])
        if self.add_extra_args:
            li.extend(["--seed=%i"%self.run_idx, "--outfile=%s"%self.outfile, "--metadata=cfg_name=%s,test_name=%s,script_name=%s"%(self.cfg_name,self.test_name,self.script_name)])
        if pipe_to_logfile == "stdout":
            pipe_str = ">"
        elif pipe_to_logfile == "all":
            pipe_str = "&>"
        elif pipe_to_logfile == "off":
            # pipe_str = ""
            pass
        if pipe_to_logfile != "off": li.append("%s %s.log\n"%(pipe_str,self.outfile))
        return " ".join(li)

def assert_script_runs_different(srs):
    scriptname2info = {sr.script_name:(sr.info,sr.run_idx,sr.command) for sr in srs}

    badkeypair = None
    valhash2key = {}
    for (k,v) in scriptname2info.items():
        valhash = cPickle.dumps(v)
        if valhash in valhash2key:
            badkeypair = (k,valhash2key[valhash])
        valhash2key[valhash] = k

    if badkeypair is not None:
        raise AssertionError(
            "Two scripts are being run with the exact same parameters: %s and %s"%badkeypair)




def prepare_dir_for_experiment(out_root,allow_continue=False):
    if osp.exists(out_root) and not allow_continue:
        yn = yes_or_no("%s exists. delete?"%out_root)
        if yn: shutil.rmtree(out_root)
        else: raise IOError

    mkdirp(out_root)


def ordered_load(stream):
    return yaml.load(stream, CustomYamlLoader)

class CustomYamlLoader(yaml.Loader):
    """
    Ordered load: http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts/21048064#21048064
    Include: http://stackoverflow.com/questions/528281/how-can-i-include-an-yaml-file-inside-another
    """
    def __init__(self, stream):
        self._root = osp.split(stream.name)[0]
        yaml.Loader.__init__(self,stream)
    def include(self, node):
        filename = osp.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, yaml.Loader)

CustomYamlLoader.add_constructor('!include', CustomYamlLoader.include)
def load_node(loader,node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))
CustomYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, load_node)

def last_if_list(l):
    if hasattr(l, "__len__"):
        return l[-1]
    else:
        return l

def extract_scalar_stats(fname, testinfo):
    hdf = h5py.File(fname,"r")
    return [last_if_list(eval(expr, dict(), dict(hdf=hdf))) for expr in testinfo["stats"].values()]

def extract_series_stats(fname, exprs):
    hdf = h5py.File(fname,"r")
    return [eval(expr, dict(), dict(hdf=hdf)) for expr in exprs]

def list_tests(testinfos):       
    for (i,test) in enumerate(testinfos):
        print "%4i    %s"%(i,test["name"])

def increase_suffix(fname):
    pat = "-([0-9]+)$"
    current_suffix_singleton = re.findall(pat, fname)
    if current_suffix_singleton:
        assert len(current_suffix_singleton) == 1
        current_suffix = current_suffix_singleton[0]
        i = int(current_suffix)
        return re.sub(pat, "-"+str(i+1), fname)
    else:
        return fname + "-1"

def test_increase_suffix():
    assert increase_suffix("/x/y/z") == "/x/y/z-1"
    assert increase_suffix("/x/y-1/z") == "/x/y-1/z-1"
    assert increase_suffix("/x/y-1/z-1") == "/x/y-1/z-2"

def get_next_suffixed_dir(dir): #pylint: disable=W0622
    n = 0
    for i in xrange(20):
        if osp.exists(dir+"-"+str(i)):
            n=i+1
    if n==0 and not osp.exists(dir):
        return dir
    else:
        return dir + "-"+str(n)



def experiment_h5name(testname, i_run):
    return testname+"_RUN%i"%i_run+".h5"


if __name__ == "__main__":
    test_increase_suffix()