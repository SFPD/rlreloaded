#!/usr/bin/env python
from control4.cloud.cloud_interface import load_cloud_config,create_cloud
import argparse
from control4.misc.console_utils import call_and_print,Popen_and_print
from control4.config import CTRL_ROOT
import os,os.path as osp




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cloud_config = load_cloud_config(provider='gce')
    cloud = create_cloud(cloud_config)

    def sync(pat="*"):
        os.chdir(osp.dirname(CTRL_ROOT))
        call_and_print("zsh -c 'ls control/**/*.py > /tmp/ctrlfiles.txt'")
        call_and_print("zsh -c 'ls control/**/*.yaml >> /tmp/ctrlfiles.txt'")
        call_and_print("zsh -c 'ls control/**/*.json >> /tmp/ctrlfiles.txt'")
        popens = []
        for d in cloud.list_instances_glob(pat):
            ip = d['networkInterfaces'][0]['accessConfigs'][0]['natIP']
            popen = Popen_and_print(
                'rsync -azv  --files-from=/tmp/ctrlfiles.txt' \
                ' -e "ssh -o UserKnownHostsFile=/dev/null -o CheckHostIP=no -o StrictHostKeyChecking=no -i /Users/xyz/.ssh/google_compute_engine -A -p 22" '\
                ' ~/Proj  xyz@%(ip)s:~'\
                %dict(ip=ip))
            popens.append(popen)
        for popen in popens:
            popen.communicate()
    def killallpython(pat="*"):
        popens = []
        for d in cloud.list_instances_glob(pat):
            ip = d['networkInterfaces'][0]['accessConfigs'][0]['natIP']
            popen = Popen_and_print("ssh -o UserKnownHostsFile=/dev/null -o CheckHostIP=no -o StrictHostKeyChecking=no -i /Users/xyz/.ssh/google_compute_engine -A -p 22 xyz@%(ip)s 'killall python'"%dict(ip=ip))
            popens.append(popen)
        for popen in popens:
            popen.communicate()
    def pull(pat="*"):
        for d in cloud.list_instances_glob(pat):
            ip = d['networkInterfaces'][0]['accessConfigs'][0]['natIP']
            call_and_print("ssh -o UserKnownHostsFile=/dev/null -o CheckHostIP=no -o StrictHostKeyChecking=no -i /Users/xyz/.ssh/google_compute_engine -A -p 22 xyz@%(ip)s 'cd ~/control; git pull; git checkout john-devel'"%dict(ip=ip),check=False)
