import zmq
from control4.misc.randomness import random_string
from control4.misc.collection_utils import chunk_slices
from control4.parallel.parallel import PoolBase,get_cpu_count
import os,time
def child_process_loop(addr):
    context = zmq.Context() #pylint: disable=E1101
    socket = context.socket(zmq.REP) #pylint: disable=E1101
    socket.bind(addr)
    while True:
        tup = socket.recv_pyobj()

        if tup[0] == "mapreduce":
            func,reducer,li = tup[1:]
            result = reducer(map(func, li))
        elif tup[0] == "apply":
            func,arg = tup[1:]
            result = func(arg)
        elif tup[0] == "exit":
            socket.send_pyobj("")
            return
        else:
            raise RuntimeError("invalid command %s"%tup[0])

        #  Send reply back to client
        socket.send_pyobj(result)




class IPCProcessPool(PoolBase):
    def __init__(self,n=-1):
        if n==-1: n = get_cpu_count()
        elif n==1: print "warning: starting pool with one process"
        elif n > 1: pass
        else: raise RuntimeError("invalid number of processes: %i"%n) 

        self.context = zmq.Context() #pylint: disable=E1101
        self.sockets = []
        self.pids = []

        prefix = random_string(12)
        pipenames = ["%s-%.4i"%(prefix,i) for i in xrange(n)]

        for pipename in pipenames:
            addr = "ipc:///tmp/%s"%pipename
            pid = os.fork()
            if pid == 0:                                
                child_process_loop(addr)
                os._exit(0) # skip all exit processing, e.g. exitfuncs, __del__ #pylint: disable=W0212
            else:
                self.pids.append(pid)
                socket = self.context.socket(zmq.REQ) #pylint: disable=E1101
                socket.connect(addr)
                self.sockets.append(socket)
    def size(self):
        return len(self.sockets)
        
    def _dispatch(self, tuples):
        assert len(tuples)>0
        n_results = len(tuples)
        assert n_results <= self.size()

        # self.log.info("sending %i messages",size)
        for (socket,tup) in zip(self.sockets,tuples):
            socket.send_pyobj(tup)

        results = [None for _ in xrange(n_results)]
        
        n_done = 0

        # self.log.info("pulling in %i messages",n_results)
        while n_done < n_results:
            for i in xrange(n_results):
                if results[i] is None:
                    try:
                        reply = self.sockets[i].recv_pyobj(zmq.DONTWAIT) #pylint: disable=E1101
                        results[i] = reply
                        n_done += 1
                    except zmq.Again: #pylint: disable=E1101
                        continue
                    # self.log.info("received %i/%i",n_done,n_results)
            time.sleep(.001)        

        return results        

    def mapreduce(self, func, reducer, items):
        return reducer(self._dispatch([("mapreduce",func,reducer,items[chunk]) for chunk in chunk_slices(len(items),self.size())]))

    def scatter(self, func, items):
        return self._dispatch([("apply",func,items[chunk]) for chunk in chunk_slices(len(items),self.size())])

    def gather(self, func, reducer, arg):
        return reducer(self._dispatch([("apply",func,arg) for _ in xrange(self.size())]))



    def __del__(self):
        for socket in self.sockets:
            socket.send_pyobj(("exit",))
        print "waiting for child processes to close..."
        for pid in self.pids:
            os.waitpid(pid,0)
        print "ok"

