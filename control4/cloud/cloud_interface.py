from fnmatch import fnmatch
from control4.misc.randomness import random_pronounceable_string
from control4.misc.func_utils import memoized

class Cloud(object):

    # Core methods that need to be defined for different providers (Google, AWS, etc)


    def start_instances(self,num_instances, params,instance_names):
        """
        params: dict with keys:
            - num_instances
            ... various provider-specific options
        """
        raise NotImplementedError

    def delete_instances(self,instance_names):
        raise NotImplementedError

    def list_instances(self):
        """
        Return a list of dicts with provider-specific info
        """
        raise NotImplementedError

    def instance_address(self,info):
        """
        Get address from info dict
        """
        raise NotImplementedError

    def instance_address_local(self,info):
        """
        Get address from info dict
        """
        raise NotImplementedError

    def instance_name(self, info):
        """
        Get name from info dict
        """
        raise NotImplementedError

    def run_shell_command(self,info,command):
        raise NotImplementedError


    # Utility methods

    def list_instances_glob(self,pat):
        return filter(lambda info: fnmatch(self.instance_name(info),pat), self.list_instances()) #pylint: disable=W0110

    def list_instances_cluster(self,cluster_name):
        return self.list_instances_glob("%s-slave*"%cluster_name)

    def get_cluster_names(self,infos=None):
        """
        Returns names
        """
        if infos is None: infos = self.list_instances()
        clusters = set()
        for info in infos:
            name = self.instance_name(info)
            if fnmatch(name, "*slave*"):
                clu = "-".join(name.split("-")[:-1])
                clusters.add(clu)

        return list(clusters)

    def get_instance_names(self,pat="*"):
        return [self.instance_name(d) for d in self.list_instances_glob(pat)]

@memoized
def load_cloud_config(**kw):
    from control4.config import load_config
    cloud_config = load_config()["cloud"]
    for (k,v) in kw.items():
        assert k in cloud_config
        cloud_config[k] = v
    return cloud_config

def create_cloud(cloud_config=None):
    if cloud_config is None:
        cloud_config = load_cloud_config()
    provider = cloud_config["provider"]
    if provider == "gce":
        from . import cloud_gce
        return cloud_gce.GCECloud()
    elif provider == "fake":
        from . import cloud_fake
        return cloud_fake.FakeCloud()    
    else:
        raise RuntimeError("Invalid cloud provider: %s"%provider)


def get_slave_names(num_instances,instance_prefix=None,start_idx=0):
    if instance_prefix is None: instance_prefix = random_pronounceable_string()
    return ["%s-slave%0.4i"%(instance_prefix,i+start_idx) for i in xrange(num_instances)]


##############
