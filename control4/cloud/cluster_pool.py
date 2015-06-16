from control4.parallel.parallel import PoolBase
from control4.misc.collection_utils import chunk_slices,concatenate
import zmq,os
import logging,sys,time

class ClusterPool(PoolBase):

    def __init__(self,cloud,cluster,start_mode="tabula_rasa",custom_command = None):
        self.sockets = []
        self.shells = []
        self.log = logging.getLogger('cluster:master')

        assert start_mode in ("tabula_rasa","the_prestige","custom")
        if start_mode == "custom": assert custom_command is not None
        self.cloud = cloud
        infos = cloud.list_instances_cluster(cluster)
        if len(infos) == 0:
            raise RuntimeError("invalid cluster %s has no instances"%cluster)
        self.context = zmq.Context() #pylint: disable=E1101

        for info in infos:
            name = cloud.instance_name(info)
            addr = cloud.instance_address(info)
            local_addr = cloud.instance_address_local(info)

            if start_mode == "tabula_rasa": command = "python $CTRL_ROOT/cloud/run_slave_loop.py %(local_addr)s"
            elif start_mode == "the_prestige": command = " ".join(["python"] + sys.argv + ["--slave_addr",local_addr,"&> /tmp/%s.log"%name])
            elif start_mode == "custom": command = custom_command
            else: raise RuntimeError
            self.shells.append(cloud.run_shell_command(info, command%dict(name=name,addr=addr,local_addr=local_addr),block=False))
            self.log.info("Connecting to %s at %s",name,addr)
            socket = self.context.socket(zmq.REQ) #pylint: disable=E1101
            socket.connect(addr)
            self.sockets.append(socket)
        print "size: %i"%self.size()
        
    def _dispatch(self, tuples):
        assert len(tuples) > 0
        n_results = len(tuples)
        assert n_results <= self.size()

        self.log.info("sending %i messages",n_results)

        for (socket,tup) in zip(self.sockets,tuples):
            socket.send_pyobj(tup)

        results = [None for _ in xrange(n_results)]
        
        n_done = 0

        self.log.info("pulling in %i messages",n_results)
        while n_done < n_results:
            for i in xrange(n_results):
                if results[i] is None:
                    try:
                        reply = self.sockets[i].recv_pyobj(zmq.DONTWAIT) #pylint: disable=E1101
                        results[i] = reply
                        n_done += 1
                        self.log.info("received %i/%i",n_done,n_results)
                    except zmq.Again: #pylint: disable=E1101
                        continue
            time.sleep(.001)        
        

        return results        

    def mapreduce(self, func, reducer, items):
        return reducer(self._dispatch([("mapreduce",func,reducer,items[chunk]) for chunk in chunk_slices(len(items),self.size())]))

    def scatter(self, func, items):
        return concatenate(self._dispatch([("scatter",func,items[chunk]) for chunk in chunk_slices(len(items),self.size())]))

    def gather(self, func, reducer, args):
        return reducer(self._dispatch([("gather",func,reducer,args) for _ in xrange(self.size())]))

    def size(self):
        raise NotImplementedError
        # return len(self.sockets) # XXX

    def close(self):
        for socket in self.sockets:
            socket.send_pyobj(("exit",))
        print "closing shells",os.getpid()
        for shell in self.shells:
            shell.close()

    def __del__(self):
        self.close()
