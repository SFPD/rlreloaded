from control4.parallel.parallel import ProcessPool,DummyPool
from control4.parallel.ipcprocesspool import IPCProcessPool
from control4.misc.console_utils import Message
from control4.misc.collection_utils import concatenate
def mapfunc1(x):
    return x**2
def mapfunc2((x,y)):
    return x*y
def reducer(li):
    return sum(li)

g_li = []
def setglobal(li):
    global g_li #pylint: disable=W0603
    g_li = li
    return len(li)
def getglobal(_):
    return g_li

def test_pool(pool):

    assert pool.size() >= 0

    for length in [2,5,97]:

        li1 = range(length)
        li2 = range(length,2*length)

        mapresult1 = [mapfunc1(x) for x in li1]
        mapreduceresult1 = reducer(mapresult1)
        mapresult2 = [mapfunc2((x,y)) for (x,y) in zip(li1,li2)]
        mapreduceresult2 = reducer(mapresult2)

        assert mapreduceresult1 == pool.mapreduce(mapfunc1,reducer,li1)
        assert mapreduceresult2 == pool.mapreduce(mapfunc2,reducer,zip(li1,li2))

        scatterresult1 = pool.scatter(setglobal, li1)
        gatherresult1 = pool.gather(getglobal,concatenate,())
        print gatherresult1,li1
        assert gatherresult1 == li1 # XXX ProcessPool gets the wrong order!

if __name__ == "__main__":
    for thunk in [lambda : IPCProcessPool(1),lambda : IPCProcessPool(4),lambda : IPCProcessPool(16),lambda:DummyPool()]:
        pool = thunk()
        with Message("testing pool: %s"%pool):
            test_pool(pool) # XXX WTF?
            pool.close()
            del pool # WTF?
    print "done"
