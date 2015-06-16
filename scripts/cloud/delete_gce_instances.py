#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pat",default="*")
args = parser.parse_args()
from fnmatch import fnmatch
from control4.cloud.cloud_interface import load_cloud_config,create_cloud
from control4.misc.console_utils import call_and_print

cloud_config = load_cloud_config(provider='gce')
gce_config = cloud_config['gce']
cloud = create_cloud(cloud_config)
infos = cloud.list_instances()
names = [cloud.instance_name(info) for info in infos if fnmatch(info["name"],args.pat)]
call_and_print("gcloud compute instances delete %s --zone %s"%(" ".join(names), gce_config['zone']))
