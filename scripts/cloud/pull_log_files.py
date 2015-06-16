#!/usr/bin/env python
from control4.cloud.cloud_interface import load_cloud_config,create_cloud
from control4.misc.console_utils import mkdirp,colorize,yes_or_no
from threading import Thread,Lock
import argparse,datetime,subprocess
from fnmatch import fnmatch
import os.path as osp, shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pat",default="*")
    parser.add_argument("--outdir",default="/tmp/gce_script_logs")
    parser.add_argument("--dry")
    parser.add_argument("--pull_h5",action="store_true")
    args = parser.parse_args()

    outdir = args.outdir or datetime.datetime.utcnow().strftime('/tmp/logfiles_%Y-%m-%d-%H-%M-%S')
    if not args.dry: 
        if osp.exists(outdir):
            if yes_or_no("%s exists. delete?"%outdir):
                shutil.rmtree(outdir)
            else:
                raise IOError
        mkdirp(outdir)

    cloud_config = load_cloud_config(provider='gce')
    gce_config = cloud_config['gce']
    cloud = create_cloud(cloud_config)
    infos = cloud.list_instances()
    names = [cloud.instance_name(info) for info in infos if fnmatch(info["name"],args.pat)]

    fnamepat = "*.h5*" if args.pull_h5 else "*.log"

    PRINT_LOCK = Lock()
    def pull_logfiles(targdir):
        cmd = "gcutil pull %s /home/xyz/data/results/*/%s %s"%(targdir,fnamepat,outdir)
        with PRINT_LOCK:
            print colorize(cmd,color='green',bold=True)
        if not args.dry: subprocess.call(cmd,shell=True)


    threads = [Thread(target=pull_logfiles, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
