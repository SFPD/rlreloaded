#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pat")
parser.add_argument("--instances",nargs="+")
args = parser.parse_args()
from control4.cloud.cloud_interface import load_cloud_config,create_cloud
from threading import Thread


cloud_config = load_cloud_config(provider='gce')
gce_config = cloud_config['gce']
cloud = create_cloud(cloud_config)
if args.pat is not None:
    infos = cloud.list_instances_glob(args.pat)
elif args.instances is not None:
    infos = cloud.list_instances()
    infos = [info for info in infos if cloud.instance_name(info) in args.instances]
else:
    raise RuntimeError("specify either --pat or --instances")


def stop(info):
    cloud.run_shell_command(info, "killall -s INT python")


threads = [Thread(target=stop, args=(info,)) for info in infos]

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()


