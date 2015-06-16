import cPickle

################################
# Basic stuff

def identity(x):
    return x

################################
# Mapreduce

def sum_reducer(it_sum):
    total_sum = None
    for vec in it_sum:
        if total_sum is None: total_sum = vec
        else: total_sum += vec
    return total_sum

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
# Caching

class once:
    def __init__(self,fn):
        self.fn = fn
        self.out = None
        self.first_time = True
    def __call__(self,*args,**kw):
        if self.first_time:
            self.out = self.fn(*args,**kw)
            self.first_time = False
        return self.out

class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args,**kw):
        key = cPickle.dumps((args,kw))
        # key = hashlib.md5(cPickle.dumps((args,kw))).hexdigest()
        try:
            return self.cache[key]
        except KeyError:
            value = self.func(*args,**kw)
            self.cache[key] = value
            return value

################################
# Functions of functions

PARTIAL_FUNCS = []
class Partial(object):
    """
    Create Partial object before forking.
    func, *args and **kws will be saved globally so they will be available to children without pickling them.
    """

    def __init__(self,func,*args,**kws):
        self.key = len(PARTIAL_FUNCS) 
        PARTIAL_FUNCS.append ( (func,args,kws)  )
    def __call__(self, *args,**kws):
        func,partial_args,partial_kws = PARTIAL_FUNCS[self.key]
        kws.update(partial_kws)
        return func( *(partial_args + args), **kws)
    @staticmethod
    def clear():
        global PARTIAL_FUNCS
        PARTIAL_FUNCS = []

class Starify(object):
    """
    Turns f(a,b,c) into f((a,b,c))
    """
    def __init__(self, f):
        self.f = f
    def __call__(self, arg):        
        return self.f(*arg)

class TwoStarify(object):
    """
    Turns f(a,b,c) into f((a,b,c))
    """
    def __init__(self, f):
        self.f = f
    def __call__(self, arg):        
        return self.f(**arg)

class Singletonize(object):
    def __init__(self,f):
        self.f=f
    def __call__(self,*args):
        return [self.f(*args)]

