import numpy as np

def chunkify(seq,chunksize):
    """
    Return a generator giving lists with length in {1,2,...,L}
    """
    assert chunksize > 0
    it = iter(seq)
    done = False
    while not done:
        batch = []
        for _ in xrange(chunksize):
            try:
                batch.append(it.next())
            except StopIteration:
                done = True
                break
        if len(batch) > 0:
            yield batch

def concatenate(li_li):
    out = []
    for li in li_li:
        out.extend(li)
    return out

def chunk_slices(n,k):
    """
    Slices that break 0:n into k chunks
    """
    if k==1: 
        return [slice(0,n)]
    elif k >= n:
        return [slice(i,i+1) for i in xrange(n)]
    else:
        edges = np.floor(np.linspace(0,n,k+1)).astype('int')
        return [slice(start,stop) for (start,stop) in zip(edges[:-1],edges[1:])]
  
def dict_update(d0,d1):
    out = d0.copy()
    out.update(d1)
    return out

def filter_dict(d, cond):
    return type(d)([(k,v) for (k,v) in d.items() if cond(k,v)])
