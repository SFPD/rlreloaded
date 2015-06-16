

def deep_getitem(d,seq):
    node = d
    for el in seq:
        node = node[el]
    return node

def deep_in(d,seq):
    try:
        deep_getitem(d,seq)
        return True
    except KeyError:
        return False


def deep_setitem(d,seq,val):
    node = d
    for el in seq[:-1]:
        node = node[el]
    node[seq[-1]] = val

def deep_list_keys(d):
    out = []
    for (k,v) in d.items():
        if isinstance(v,dict):
            for kseq in deep_list_keys(v):
                out.append([k]+kseq)
        else:
            out.append([k])
    return out