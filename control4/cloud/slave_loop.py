# from control3.parallel import MyPool
import zmq
import logging
from control4.config import setup_logging
import multiprocessing
from control4.parallel.parallel import ProcessPool

def slave_loop(addr):    
    setup_logging()
    log = logging.getLogger("cluster:slave")
    log.setLevel(logging.INFO)

    log.info("entering slave loop, with address %s. In process %s",addr,multiprocessing.current_process().name)
    pool = ProcessPool(-1) 
    log.info("made a pool with size %i",pool.size())

    context = zmq.Context() #pylint: disable=E1101
    socket = context.socket(zmq.REP) #pylint: disable=E1101
    socket.bind(addr)

    i_msg = 0
    while True:
        #  Wait for next request from client
        log.info("recv...")
        tup = socket.recv_pyobj()

        log.info("doing job %i of type %s",i_msg, tup[0])
        if tup[0] == "mapreduce":
            func,reducer,li = tup[1:]
            result = pool.mapreduce(func, reducer, li)
        elif tup[0] == "scatter":
            func,li = tup[1:]
            result = pool.scatter(func,li)
        elif tup[0] == "gather":
            func,reducer,args = tup[1:]
            result = pool.gather(func,reducer,args)
        elif tup[0] == "exit":
            log.info("exiting loop")
            exit(0)

        #  Send reply back to client
        # log.info("sending")
        socket.send_pyobj(result)

        i_msg += 1
