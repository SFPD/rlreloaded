#!/usr/bin/env python
from control4.cloud.cloud_interface import load_cloud_config,create_cloud,get_slave_names
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_slaves",action="store_true")
    parser.add_argument("--slave_prefix")
    parser.add_argument("--num_slaves",type=int)
    parser.add_argument("--start_master",action="store_true")
    parser.add_argument("--master_name",type=str,default="master")
    parser.add_argument("--slave_start_idx",type=int,default=0)
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--dry",action="store_true")
    args = parser.parse_args()

    cloud_config = load_cloud_config()
    cloud = create_cloud(cloud_config)

    instance_names = []
    if args.start_slaves:
        instance_names.extend(get_slave_names(args.num_slaves,instance_prefix=args.slave_prefix,start_idx=args.slave_start_idx))
    if args.start_master:
        instance_names.append(args.master_name)
    if len(instance_names)>0: cloud.start_instances(instance_names=instance_names,dry=args.dry)
    if args.list:
        print "**** ALL INSTANCES ****"
        infos = cloud.list_instances()
        for info in infos:
            print cloud.instance_name(info), cloud.instance_address(info)
        print "**** CLUSTERS ****"        
        cluster_names = cloud.get_cluster_names(infos)
        "\n".join(cluster_names)
        
    # cloud.start_instances()



if __name__ == "__main__":
    main()