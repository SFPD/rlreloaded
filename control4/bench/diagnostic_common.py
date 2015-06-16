from control4.algs.alg_params import string2dict
import h5py
import numpy as np
from tabulate import tabulate

def pad_ends(lis):
    maxlen = max(len(li) for li in lis)
    def pad(li,n):
        li = list(li)
        return li + [li[-1]]*(n-len(li))
    return [pad(li,maxlen) for li in lis]

def load_diagnostic_info(hdf):
    metadata = string2dict(hdf["params"]["metadata"].value)

    rows = []
    i_run = hdf["params"]["seed"].value
    cfg_name = str(metadata["cfg_name"])
    test_name = metadata["test_name"]
    script_name = metadata["script_name"]

    for (stat_name,ts) in hdf["diagnostics"].items():
        rows.append((cfg_name, test_name, script_name, stat_name, i_run, ts.value))
    columns = ["cfg_name","test_name","script_name","stat_name","i_run","timeseries"]
    return rows, columns


def load_hdfs_as_dataframe(h5files):
    from pandas import DataFrame
    allrows = []
    for h5file in h5files:
        try:
            hdf = h5py.File(h5file,'r')
            rows,columns = load_diagnostic_info(hdf)
            allrows.extend(rows)
            hdf.close()
        except (IOError,RuntimeError):
            print "failed to open %s"%h5file
    df = DataFrame(allrows, columns = columns)
    return df

def compute_mean_std_across_runs(df):
    from pandas import DataFrame

    fields = [col_name for (col_name,example) in zip(df.columns,df.irow(0)) if isinstance(example,(str,unicode))]


    newrows = []
    for (_,grp) in df.groupby(fields):
        firstrow = grp.irow(0)
        newrow = firstrow.tolist()[:-1]
        newrow.append(np.mean(pad_ends(grp["timeseries"]),axis=0))
        newrow.append(np.std(pad_ends(grp["timeseries"]),axis=0)/np.sqrt(max(len(grp)-1,1)))
        newrows.append(newrow)
        
    df = DataFrame(newrows, columns=list(df.columns[:-1])+["timeseries","timeseries_stderr"])
    return df


def disp_dict_as_3d_array(d, key0=None, key1=None, key2=None,use_numeric_keys=False):
    """
    Assume dict keys are tuples
    Take it as a 3d array and print
    """
    set0 = set()
    set1 = set()
    set2 = set()
    for (k0,k1,k2) in d:
        set0.add(k0)
        set1.add(k1)
        set2.add(k2)
    arr = np.zeros((len(set0),len(set1),len(set2)),object)
    keys0 = sorted(set0,key0)
    keys1 = sorted(set1,key1)
    keys2 = sorted(set2,key2)
    for ((k0,k1,k2),v) in d.items():
        arr[keys0.index(k0),keys1.index(k1),keys2.index(k2)] = v
    header_keys = map(str,range(len(keys2))) if use_numeric_keys else keys2        
    for (k0,tabledata) in zip(keys0,arr):
        title = "***** %s *****"%k0
        print
        print "*"*len(title)
        print title
        print "*"*len(title)
        print tabulate(  [[str(k1)]+row.tolist() for (k1,row) in zip(keys1,tabledata)]  ,headers=['cfg']+header_keys)
    if use_numeric_keys:
        for (i,k2) in enumerate(keys2):
            print "%i: %s"%(i,k2)


def disp_dict_as_2d_array(d, key0=None, key1=None, key2=None,use_numeric_keys=False): #pylint: disable=W0613
    """
    Assume dict keys are tuples
    Take it as a 3d array and print
    """
    set0 = set()
    set1 = set()
    for (k0,k1) in d:
        set0.add(k0)
        set1.add(k1)
    arr = np.zeros((len(set0),len(set1)),object)
    keys0 = sorted(set0,key0)
    keys1 = sorted(set1,key1)
    for ((k0,k1),v) in d.items():
        arr[keys0.index(k0),keys1.index(k1)] = v
    header_keys = map(str,range(len(keys1))) if use_numeric_keys else keys1
    print tabulate(  [[str(k0)]+row.tolist() for (k0,row) in zip(keys0,arr)]  ,headers=['']+header_keys)
    if use_numeric_keys:
        for (i,k1) in enumerate(keys1):
            print "%i: %s"%(i,k1)



