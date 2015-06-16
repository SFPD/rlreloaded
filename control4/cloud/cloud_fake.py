from control4.cloud.cloud_interface import Cloud
from control4.misc.console_utils import mkdirp,Popen_and_print
import json, os.path as osp
from glob import glob

class FakeShell(object):
    def __init__(self,popen):
        self.popen = popen
    def close(self):
        self.popen.terminate()
        self.popen.wait()

class FakeCloud(Cloud):



    def start_instances(self, instance_names, dry=False): #pylint: disable=W0613
        infos = [{"addr":"ipc:///tmp/%s"%name,"name":name} for name in instance_names]

        fakecluster_dir = self._fakecluster_dir()
        mkdirp(fakecluster_dir)
        fname = osp.join(fakecluster_dir,"%s.json"%("-".join(instance_names[0].split("-")[:-1])))

        with open(fname,'w') as fh:
            json.dump(infos,fh)

        print "dumped fake instance info to %s"%fname

    def list_instances(self):
        results = []
        for fname in glob(osp.join(self._fakecluster_dir(),"*.json")):
            with open(fname,'r') as fh:
                infos = json.load(fh)
                results.extend(infos)
        return results
        

    def instance_address(self,info):
        return info['addr']

    def instance_address_local(self,info):
        return self.instance_address(info)

    def instance_name(self,info):
        return info['name']

    def run_shell_command(self,info,cmd,block=True):
        popen = Popen_and_print(cmd)
        if block: popen.communicate()
        else: return FakeShell(popen)

    def delete_instances(self,instance_names):
        raise NotImplementedError



    def _fakecluster_dir(self):
        return "/tmp/fakeclusters"

