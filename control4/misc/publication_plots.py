from scipy import stats
from scipy.interpolate import interp1d
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import os.path as osp
from glob import glob
from control4.bench.diagnostic_common import load_hdfs_as_dataframe,compute_mean_std_across_runs

pd.options.display.mpl_style = 'default'

def make_tableau20():
    # These are the "Tableau 20" colors as RGB.  
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
    for i in range(len(tableau20)):  
        r, g, b = tableau20[i]  
        tableau20[i] = (r / 255., g / 255., b / 255.)

    for (a, b) in [(1, 4), (3, 16)]:
        tmp = tableau20[b]
        tableau20[b] = tableau20[a]
        tableau20[a] = tmp
    return tableau20

colors = make_tableau20()

def plot_scatter(data, ax, text, color, span=10, method='emwa'):
    from pandas.stats.moments import ewma
    data = ewma(data,span=span)

    ax.plot(data, color=color, linewidth=1, label=text)

def get_data(fname, objective):
    h5files = glob(fname)
    df = load_hdfs_as_dataframe(h5files)
    df = compute_mean_std_across_runs(df)
    rows =  df["timeseries"][df["stat_name"] == objective]
    assert len(rows)==1
    print rows.irow(0)
    return rows.irow(0)

def plot_dataset(dataset, datadir, ax):
    datalist = dataset['comparisons']
    data = []

    missing = []
    for (fname, method) in datalist:
        fullpath = osp.join(datadir,fname)
        if len(glob(fullpath))==0:
            print "file %s does not exist!"%fullpath
            missing.append(fullpath)
    if len(missing)>0:
        raise IOError("missing %s"%" ".join(missing))
    
    i = 0
    for (fname, method) in datalist:
        fname = osp.join(datadir, fname)
        i += 1
        print "processing ", fname
        data = get_data(fname, dataset['objective'])
        plot_scatter(data, ax, method, colors[i])

    ax.set_xlabel("number of policy iterations")
    ax.set_ylabel(dataset["ylabel"])
    ax.set_title(dataset['title'])
    ax.grid(True)
    if dataset['title'] in ('2D walker (velocity)','qbert'):
        ax.legend(loc=3,fontsize=18)
    else:
        ax.legend(fontsize=10)