from control4.cloud.cloud_interface import *
from control4.misc.console_utils import maybe_call_and_print,check_output_and_print,colorize,call_and_print
from control4.config import load_config

import threading
import time,json,re
import pexpect
import os.path as osp
import tempfile
import subprocess

# XXX terminal output from shells is odd

RUN_COMMANDS_SCRIPT_TEMPLATE = """
#!/bin/bash

%(basic_setup)s

# Actually run commands
cd
%(cmds_str)s

# Upload files
gsutil cp -R $CTRL_DATA/results/* %(new_data_bucket)s/results/
gsutil cp /var/log/startupscript.log %(new_data_bucket)s/logs/%(instance_name)s.log
"""

BASIC_SETUP_TEMPLATE = """

cd ~/control
git pull
git checkout %(branch_name)s

cd ~/control/cpp/3rdparty/mjc2
git pull

cd ~/Theano
git pull

cd ~/build/control
make -j

"""

STARTUP_SCRIPT_TEMPLATE = """

#!/bin/bash
if [[ $(id -u) -eq 0 ]];
then
    sudo cp /var/run/google.startup.script /tmp/blah.bash
    sudo chown %(image_username)s /tmp/blah.bash
    sudo -u %(image_username)s bash /tmp/blah.bash
else
    # bash_profile and bashrc haven't been run yet since this is a root shell
    source ~/.bashrc

    %(run_commands)s

    # Maybe delete instance
    %(maybe_delete_str)s 

fi
"""

PRINT_LOCK = threading.Lock()    

def indent(text,n_tabs):
    space = " "*(4*n_tabs)
    return "\n".join([space + line for line in text.split("\n")])

# TODO delete startup scripts eventually
class GCECloud(Cloud):
    def __init__(self):
        pass

    def start_instances(self, instance_names, dry=False):
        config = load_config()

        threads = []
        for instance_name in instance_names:
            d = dict(
                branch_name = config["experiments"]["branch_name"],
                image_name = config["cloud"]["gce"]["image_name"],
                image_username = config["cloud"]["gce"]["image_username"],
                instance_name = instance_name,
                machine_type = config["cloud"]["gce"]["machine_type"],
                project = config["cloud"]["gce"]["project"],
                zone = config["cloud"]["gce"]["zone"],
                maybe_delete_str = "",                
            )
            d["run_commands"] = indent(BASIC_SETUP_TEMPLATE%d,1)
            startup_script_str =  STARTUP_SCRIPT_TEMPLATE%d
            with tempfile.NamedTemporaryFile(delete=False) as fh:
                fh.write(startup_script_str)
            d["startup_script_fname"] = fh.name


            startcmd = ('gcloud compute instances create'
                ' %(instance_name)s'
                ' --boot-disk-type "pd-standard"'
                ' --format json'
                ' --image "https://www.googleapis.com/compute/v1/projects/%(project)s/global/images/%(image_name)s"'
                ' --scopes "https://www.googleapis.com/auth/devstorage.read_only"'
                ' --machine-type %(machine_type)s'
                ' --maintenance-policy "MIGRATE"'
                ' --metadata-from-file startup-script=%(startup_script_fname)s'
                ' --network "default"'
                ' --project "%(project)s"'
                ' --zone "%(zone)s"'
                % d
            )
            threads.append(threading.Thread(target=maybe_call_and_print,args=(startcmd,dry)))
        for thread in threads:
            thread.start()
            time.sleep(.25)
        for thread in threads:
            thread.join()
        
    def delete_instances(self,instance_names):
        gce_config = load_cloud_config()["gce"]
        cmd = "gcloud compute instances delete %(instances)s --zone %(zone)s "%dict(instances=" ".join(instance_names),zone=gce_config['zone'],project=gce_config['project'])
        call_and_print(cmd)

    def list_instances(self):
        config = load_cloud_config()["gce"]
        cmd = "gcloud compute instances list --format json --zone %(zone)s"%dict(zone=config['zone'],project=config['project'])
        output = check_output_and_print(cmd)
        return json.loads(output)

    def instance_name(self,info):
        return info['name']

    def instance_address(self,info):
        return "tcp://%s:5555"%info["name"]

    def instance_address_local(self,info):
        return "tcp://*:5555"

    def run_shell_command(self,info,cmd,block=True):
        name = self.instance_name(info)        
        shell = get_gce_shell(name)
        shell.runcmd(cmd,block=block)
        if not block:
            return shell
        else:
            shell.close()
            return None


    def _gen_cmd_str(self,cmds):
        cmds_str = "" #pylint: disable=W0612
        # XXX maybe we should make sure directory structure exists in the script itself
        for cmd in cmds:
            local_fname, = re.findall(r"--outfile=(\S+)",cmd)
            cmds_str += "mkdir -p %s\n"%(osp.dirname(local_fname))
            cmds_str += cmd
        return cmds_str

    def _gen_bash_script(self,cmds,instance_name):
        config = load_config()

        return RUN_COMMANDS_SCRIPT_TEMPLATE%dict(
            branch_name = config["experiments"]["branch_name"],
            cmds_str = self._gen_cmd_str(cmds),
            instance_name = instance_name,
            new_data_bucket = config["cloud"]["gce"]["new_data_bucket"],
            basic_setup = BASIC_SETUP_TEMPLATE%config["experiments"]
        )

    # def run_commands_on_existing_instance(self,cmds, instance_name,dry=False,keep_instance=False):
    #     better to scp script and then run with nohup
    #     script_str = self._gen_bash_script(cmds, instance_name)
    #     with tempfile.NamedTemporaryFile() as fh:
    #         startup_script_fname=fh.name
    #         print "running script: "
    #         print "******************"
    #         print colorize(script_str,"green")
    #         print "******************"
    #         fh.write(script_str)
    #         cmd = "gcutil ssh %(instance_name)s 'bash -s' < %(local_fname)s"%(instance_name,local_fname)
    #     maybe_call_and_print(cmd,dry)


    def run_commands_on_fresh_instance(self,cmds,instance_name,dry=False,keep_instance=False):
        config = load_config()

        d = dict(
            branch_name = config["experiments"]["branch_name"],
            image_name = config["cloud"]["gce"]["image_name"],
            image_username = config["cloud"]["gce"]["image_username"],
            instance_name = instance_name,
            machine_type = config["cloud"]["gce"]["machine_type"],
            new_data_bucket = config["cloud"]["gce"]["new_data_bucket"],
            project = config["cloud"]["gce"]["project"],
            run_commands = indent(self._gen_bash_script(cmds,instance_name),1),
            zone = config["cloud"]["gce"]["zone"],
        )
        d["maybe_delete_str"] = "" if keep_instance else "gcutil deleteinstance --force --delete_boot_pd %(instance_name)s --project=%(project)s --zone=%(zone)s"%d #pylint: disable=W0612
        startup_script_str =  STARTUP_SCRIPT_TEMPLATE%d
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(startup_script_str)

        d["startup_script_fname"] = fh.name
        
        startcmd = ('gcloud compute instances create'
            ' %(instance_name)s'
            ' --boot-disk-type "pd-standard"'
            ' --format json'
            ' --image "https://www.googleapis.com/compute/v1/projects/%(project)s/global/images/%(image_name)s"'
            ' --scopes "https://www.googleapis.com/auth/devstorage.read_only"'
            ' --machine-type %(machine_type)s'
            ' --maintenance-policy "MIGRATE"'
            ' --metadata-from-file startup-script=%(startup_script_fname)s'
            ' --network "default"'
            ' --project "%(project)s"'
            ' --zone "%(zone)s"'
            %d)

        with PRINT_LOCK:
            print "**************************************"
            print "writing startup script to %s"%fh.name        
            print "startup script:"
            print colorize(startup_script_str,'green')
            print "start command:"
            print colorize(startcmd,'green')
            print "**************************************"

        if not dry:
            subprocess.check_call(startcmd,shell=True)


### Messy GCE auxililliary stuff

class AbstractShell(object):
    def runcmd(self, cmd, block):
        raise NotImplementedError
    def is_busy(self):
        raise NotImplementedError
    def close(self):
        raise NotImplementedError
    def __del__(self):
        self.close()


DEFAULT_SHELL_PROMPT = r"\S{1,10}@\S{1,20}\:\~\S{1,30}\$"


class PexpectShell(AbstractShell):
    def __init__(self, child, disp_prefix="",shell_prompt=None):
        self.shell_prompt = DEFAULT_SHELL_PROMPT if shell_prompt is None else shell_prompt
        self.child = child
        self.busy = False
        self.disp_prefix = disp_prefix
    def runcmd(self, cmd, block=True):
        self.busy = True
        print colorize(self.disp_prefix+cmd,'yellow')
        self.child.sendline(cmd)
        if block:
            self.child.expect(self.shell_prompt,timeout=9999999)
            print colorize(self.child.before,'magenta')
            self.busy = False
    def update(self):
        if self.busy:
            try:
                self.child.expect(self.shell_prompt,timeout=0)
                print colorize(self.child.before,'magenta')
                self.busy = False
            except pexpect.TIMEOUT:
                # print colorize(self.child.before,'magenta')
                pass
    def close(self):
        self.update()
        if self.is_busy():
            self.child.sendcontrol('c')
        self.child.close()
    def is_busy(self):
        return self.busy
    def join(self):
        while self.busy:
            self.update()
            time.sleep(.1)


def get_gce_shell(name):
    config = load_cloud_config()["gce"]
    username = config['image_username']
    for _i_try in xrange(10):
        cmd = "gcloud compute ssh %(name)s --zone %(zone)s --project %(project)s"%dict(name=name,zone=config['zone'],project=config['project'])
        print colorize(cmd, 'green')
        p = pexpect.spawn(cmd)
        shell_prompt = username + r"@\S{1,30}\:\~\S{0,30}\$"

        try:
            p.expect(shell_prompt, timeout=6)
            return PexpectShell(p, "" if name is None else (name+": "),shell_prompt=shell_prompt)
        except pexpect.TIMEOUT:
            p.close()
            print "ssh timed out. trying again"
            time.sleep(2)
        except pexpect.EOF:
            print "ssh got eof. trying again"
            try: 
                p.close()
            except pexpect.ExceptionPexpect:
                print colorize("pexpect couldn't close child", "red")
            time.sleep(5)
    raise ValueError("couldn't connect by ssh")

