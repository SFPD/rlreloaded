import cPickle,numpy as np

def pkldump(hdf, path, o):
    s = cPickle.dumps(o, -1)
    setitem_maybe_compressed(hdf,path,np.ndarray(shape=len(s), dtype='uint8', buffer=s))

def pklload(hdf, path):
    s = hdf[path].value.tostring()
    return cPickle.loads(s)

def setitem_maybe_compressed(grp,key,val):
    """
    Don't compress if it contains float data
    """
    if val.dtype.kind == 'f':
        if val.size == 0:
            grp.create_dataset(key, (), dtype='float32') 
        else:
            grp.create_dataset(key,data=val)
    else:
        if val.size == 0:
            grp.create_dataset(key, (), dtype='float32') 
        else:
            grp.create_dataset(key, data=val, compression='gzip', compression_opts=9)
