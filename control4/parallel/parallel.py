import multiprocessing,sys,time
import traceback
# from control3.common_util import chunk_slices,concatenate
from control4.misc.func_utils import Starify,TwoStarify,Singletonize
from control4.misc.collection_utils import concatenate,chunk_slices


class PoolBase(object):
    """
    Abstract class for process/thread pool
    "reducer" is assumed to have type [M] -> M and be associative    
    """

    #### Required primitive methods ####
    def size(self):
        """
        Number of processes
        """
        raise NotImplementedError
    def mapreduce(self, func, reducer, iterable):
        """
        func        :: a -> b
        reducer     :: [b] -> b
        iterable    :: [a] 

        returns     :: b

        for example,
        [a0,a1,a2)]-> reducer([f(a0),f(a1),f(a2)])
        """
        raise NotImplementedError        
    def scatter(self, func, iterable):
        """
        func        :: [a] -> [b]
        iterable    :: [a]

        returns     :: [b]

        Split up data and apply function at leaf
        Pseudocode:
            if is_leaf: return [func(iterable)]
            else: return concatenate([child.scatter(func,iterable[chunk] for chunk in chunkify(iterable))])
        """
        raise NotImplementedError
    def gather(self, func, reducer, arg):
        """
        func        :: a -> b
        reducer     :: [b] -> b
        arg         :: a

        returns     :: b

        Apply func at each leaf process
        Then apply reducer

        Pseudocode:
            if is_leaf: return func(arg)
            else: return reduce([child.gather,reducer,arg) for child in children])
        """
        raise NotImplementedError
    #### End primitives ####


    #### Utility methods ####
    def map(self, func, iterable):
        return self.mapreduce(Singletonize(func), concatenate, iterable)
    def apply(self, func, arg):
        return self.gather(Singletonize(func), concatenate, arg)
    def starmap(self, func, iterable):
        """
        [(a0,b0,c0),(a1,b1,c1),...] -> [f(a0,b0,c0),f(a1,b1,c1),...]
        """
        return self.map(Starify(func), iterable)
    def twostarmap(self, func, iterable):
        """
        [dict(a="foo0",b="bar0"),dict(a="foo1",b="bar1"),...] -> [f(a=foo0,b=bar0),f(a=foo1,b=bar1),...]
        """
        return self.map(TwoStarify(func), iterable)
    def map2(self, func, fixedargs, iterable):
        return self.starmap(func, [fixedargs + (el,) for el in iterable] )
    def close(self):
        return


class ProcessPool(PoolBase):

    counter = multiprocessing.Value('i', 0, lock=True)

    def __init__(self, n_processes):
        """
        n_processes == -1 -> use cpu_count
        otherwise it should be an int >= 1
        """
        if n_processes == -1:
            if sys.platform == "darwin":
                n_processes = 4
            else:
                n_processes = get_cpu_count()
        else:
            assert n_processes > 0
        self.pool = multiprocessing.Pool(processes=n_processes)
        self.n_processes = n_processes
        self._closed = False
    def mapreduce(self, func, reducer, iterable):
        return reducer(self.pool.map(ProcessPoolFuncWrapper(func,False),iterable))
    def scatter(self, func, iterable):
        ProcessPool.counter.value = 0
        return self.pool.map(ProcessPoolFuncWrapper(func,True,self.n_processes), [iterable[sli] for sli in chunk_slices(len(iterable),self.size())])
    def gather(self, func, reducer, arg):
        ProcessPool.counter.value = 0
        return reducer(self.pool.map(ProcessPoolFuncWrapper(func,True,self.n_processes), (arg for _ in xrange(self.size()))))
    def size(self):
        return self.n_processes
    def close(self):
        self.pool.close()
        self.pool.join()
        self._closed = True


class DummyPool(PoolBase):
    """
    Just use current process
    """
    def mapreduce(self, func, reducer, iterable):
        return reducer(map(func, iterable))
    def scatter(self, func, iterable):
        return [func(iterable)]
    def gather(self, func, reducer, arg):
        return func(arg)
    def size(self):
        return 1

################################

class ProcessPoolFuncWrapper(object):
    """ Wraps a function to make it exception with full traceback in
        their representation.
        Useful for parallel computing with multiprocessing, for which
        exceptions cannot be captured.
    """
    def __init__(self, func,delay,n_processes=None):
        self.func = func
        self.delay = delay
        self.n_processes = n_processes

    def __call__(self, *args, **kwargs):
        try:
            if self.delay:
                with ProcessPool.counter.get_lock():
                    ProcessPool.counter.value += 1
                out = self.func(*args, **kwargs)
                while ProcessPool.counter < self.n_processes:
                    time.sleep(.001)
            else:
                out = self.func(*args, **kwargs)
            return out
        except KeyboardInterrupt:
            # We capture the KeyboardInterrupt and reraise it as
            # something different, as multiprocessing does not
            # interrupt processing for a KeyboardInterrupt
            print "got keyboardinterrupt"
            raise WorkerInterrupt
        except Exception:
            traceback.print_exc()
            raise WorkerException


# from joblib/parallel.py
class WorkerInterrupt(Exception):
    """ An exception that is not KeyboardInterrupt to allow subprocesses
        to be interrupted.
    """
    pass
class WorkerException(Exception):
    """ Worker hit an exception
    """
    pass



def get_cpu_count():
    """
    want actual number of CPUs, ignoring hyperthreading
    """
    if sys.platform == "darwin":
        return 4
    else:
        return multiprocessing.cpu_count()

def sum_count_reducer(it_sum_count):    
    total_sum = None
    total_count = 0
    for (vec,count) in it_sum_count:
        if total_sum is None:
            if vec is None:
                pass
            else:
                total_sum = vec.copy()
                total_count += count
        else: # total sum is not None
            if vec is None:
                pass
            else:
                total_sum += vec
                total_count += count

    return (total_sum, total_count)


################################

testglobal = 0
def testset(x):
    global testglobal #pylint: disable=W0603
    testglobal = x
def testget(_):
    time.sleep(.1)
    return (testglobal,multiprocessing.current_process().name)


def test():
    pp = ProcessPool(2)
    pp.apply(testset, (42,))
    print pp.map(testget, xrange(10))

def testf(x):
    asdf #pylint: disable=E0602,W0104
    return x**2
def testsafefn():
    pool = ProcessPool(1)
    try:
        pool.map(testf,range(10))
    except WorkerException:
        pass



if __name__ == "__main__":
    print "__main__ in", multiprocessing.current_process()
    test()
    # testsafefn()
