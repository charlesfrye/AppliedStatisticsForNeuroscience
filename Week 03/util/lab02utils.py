import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats
from IPython.core.display import HTML
import numpy as np

def formatDataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))

def plotDistributions():
    norm = scipy.stats.norm()
    laplace = scipy.stats.laplace()
    xs = np.linspace(-5,5,1000)
    plt.plot(xs,[norm.pdf(x) for x in xs],linewidth=4,label='Gaussian');
    plt.plot(xs,[laplace.pdf(x) for x in xs],linewidth=4,label='Laplace');
    plt.legend();

def generateDataset(N=15,distribution='gauss'):
    """
    Produce a dataset of size N with a given distribution.
    Distribution must be either Gaussian or Laplacian.
    Location and scale parameters are fixed at 0 and 4.

    Parameters
    ----------
    N            : integer, how many samples to draw
    distribution : string, either 'gauss' or 'laplace'

    Returns
    -------
    A numpy array of length N an error
    """

    if distribution == 'laplace':
        return np.random.laplace(scale=1,size=N)
    elif distribution == 'gauss':
        return np.random.normal(scale=4,size=N)
    else:
        raise ValueError("distribution must be laplace or gauss")
        return

def plotExperiment(means,medians):
    styleDictionary = { 'linewidth': 4,
                          'edgecolor': 'white',
                           'normed':True}
    mn = min(min(means),min(medians))
    mx = max(max(medians),max(means))
    bins = np.linspace(mn,mx,25)

    sns.distplot(means,label='means',kde=False,bins=bins,hist_kws=styleDictionary);
    sns.distplot(medians,label='medians',kde=False,bins=bins,hist_kws=styleDictionary);

    plt.vlines(0,ymin=0,ymax = plt.ylim()[1],
           linewidth=4,linestyle = '--',
           label='true value')

    plt.legend()

    return

def plotConfidenceIntervals(intervals,trueMean):
    numIntervals = len(intervals)
    mins = [interval[0] for interval in intervals]
    maxs = [interval[1] for interval in intervals]
    for idx,interval in enumerate(intervals):
        if interval[0] < trueMean < interval[1]:
            color = 'black'
        else:
            color = 'red'
        plt.hlines(idx,interval[0],interval[1],color=color)
        plt.xlim([-4,4])
    return

def plotGroups(df,x,by):
    facets = sns.FacetGrid(df,hue=by,size=6)
    facets.map(sns.distplot,x,
               kde=False,
               hist_kws={"histtype":"stepfilled",
                        "alpha":0.5,
                        "normed":True});
    plt.ylabel('p')
    plt.gca().set_title('Histograms')
    plt.gca().legend()
    return plt.gca()

def generateDataset2(N=1000):
    """
    generates two normally distributed datasets
    of size N each,
    their means differing by one-tenth their variance,
    and returns them
    organized into a pandas dataframe
    with two columns:
        score, floats, the random normal values
        group, strings, the identity of the dataset ("control" or "treatment"
    """
    group1 = np.random.normal(0,1,size = N)
    group2 = np.random.normal(0.1,1,size = N)

    data = np.concatenate((group1,group2))
    df = pd.DataFrame.from_dict({'score':data,
                                 'group':['control']*N+['treatment']*N})
    return df

def addAxisLine(axis):
    """
    adds a dashed horizontal grey line at height 0
    """
    xlims = axis.get_xlim()
    axis.hlines(0,xlims[0],xlims[1],
           color='grey',linestyle='--',linewidth=4);
    return

def plotSDbars(data):
    """
    plot standard deviation bars using
    pandas group-by and the pyplot error bar function
    """
    grouped = data.groupby("group")
    means = grouped.aggregate(np.mean).score
    sds = grouped.aggregate(np.std).score

    plt.errorbar([0,1],means,yerr=sds,
             linestyle='None',color=[.26]*3+[0.9],
             linewidth=4,capsize=12,capthick=4);
