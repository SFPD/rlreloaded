import sys,subprocess,time,os.path as osp,os

def yes_or_no(question):
    assert isinstance(question,str) or isinstance(question,unicode)
    while True:
        sys.stderr.write(question + " (y/n): ")
        yn = raw_input()
        if yn=='y': return True
        elif yn=='n': return False

def mkdirp(d):
    d = osp.expandvars(d)
    if not osp.exists(d):
        os.makedirs(d)

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def call_and_print(cmd,color='green',check=True):
    print colorize(cmd, color, bold=True)
    if check: subprocess.check_call(cmd, shell=True)
    else: return subprocess.call(cmd, shell=True)

def check_output_and_print(cmd,color='green'):
    print colorize(cmd, color, bold=True)
    return subprocess.check_output(cmd, shell=True)

def Popen_and_print(cmd, color='green', **kw):
    print colorize(cmd, color, bold=True)
    return subprocess.Popen(cmd, shell=True, **kw)

def maybe_call_and_print(cmd, dry, **kwargs):
    if dry:
        print colorize("(dry) " + cmd,'green')
    else:
        call_and_print(cmd,**kwargs)    

MESSAGE_DEPTH = 0
class Message(object):
    def __init__(self, msg):
        self.msg = msg
    def __enter__(self):
        global MESSAGE_DEPTH #pylint: disable=W0603
        print colorize('\t'*MESSAGE_DEPTH + '=: ' + self.msg,'magenta')
        self.tstart = time.time()
        MESSAGE_DEPTH += 1
    def __exit__(self, etype, *args):
        global MESSAGE_DEPTH #pylint: disable=W0603
        MESSAGE_DEPTH -= 1
        maybe_exc = "" if etype is None else " (with exception)"
        print colorize('\t'*MESSAGE_DEPTH + "done%s in %.3f seconds"%(maybe_exc, time.time() - self.tstart), 'magenta')


class Timers(object):
    def __init__(self):
        self.key2tc = {}

    def wrap(self,fn,key):
        assert key not in self.key2tc
        self.key2tc[key] = (0,0)
        def timedfn(*args):
            tstart = time.time()
            out = fn(*args)
            (told,cold) = self.key2tc[key]
            dt = time.time() - tstart
            self.key2tc[key] = (told+dt, cold+1)
            return out
        return timedfn

    def stopwatch(self, key):
        if key not in self.key2tc:
            self.key2tc[key]=(0,0)
        class ScopedTimer(object):
            def __enter__(self):
                self.tstart = time.time()
            def __exit__(self1,*_args): #pylint: disable=E0213
                told,cold = self.key2tc[key]
                dt = time.time()-self1.tstart
                self.key2tc[key] = (told + dt, cold + 1)
        return ScopedTimer()

    def disp(self, s="elapsed time"):
        header = "******** %s ********"%s
        print header
        rows = [(key, t, c, t/c) for (key,(t,c)) in self.key2tc.items() if c>0]
        from tabulate import tabulate
        print tabulate(rows, headers=["desc","total","count","per"])


class InDir(object):
    """
    Enter directory when scope is entered
    Leave it when scope is exited
    """
    def __init__(self,pth):
        self.pth = pth
    def __enter__(self):
        self.curpth = os.getcwd()
        os.chdir(self.pth)
    def __exit__(self,*_):
        os.chdir(self.curpth)

def dict_as_table(d):
    from tabulate import tabulate
    def fmt(v):
        s = str(v)
        if len(s) > 20:
            s = s[:17]+"..."
        return s
    return tabulate([(k,fmt(v)) for (k,v) in sorted(d.items())])
