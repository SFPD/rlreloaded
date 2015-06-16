#!/usr/bin/env python
from cloud.cloud_interface import *
import os,atexit
import os.path as osp
from cloud.cluster_pool import ClusterPool
import time
from control3.common import setup_logging
import math

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider",choices=["fake","gce"],default="fake")
    parser.add_argument("--cluster")
    args = parser.parse_args()

    setup_logging()

    if args.provider == "fake":
        cloud_config = load_cloud_config(provider="fake")
        cloud = create_cloud()
        prefix = "testmap"
        @atexit.register
        def cleanup():
            print "cleanup"
            fname = "/tmp/fakeclusters/testmap.json"
            if osp.exists(fname):
                os.unlink(fname)
        cloud.start_instances(instance_names=get_slave_names(3,instance_prefix=prefix))
        cluster=prefix
        infos = cloud.list_instances_cluster(cluster)
        pool = ClusterPool(cloud, prefix, start_mode = "tabula_rasa")

    elif args.provider == "gce":
        cloud_config = load_cloud_config(provider="gce")
        cloud = create_cloud(cloud_config)
        if args.cluster is None:
            cluster_names = cloud.get_cluster_names()
            cluster = cluster_names[0]
            assert len(cluster_names) == 1 
        else:
            cluster = args.cluster

        pool = ClusterPool(cloud, cluster, start_mode = "tabula_rasa")
    else:
        raise RuntimeError




    time.sleep(1)

    for i in xrange(10):
        result = pool.map(math.sqrt, range(100))
    print "result:", result

