from random import shuffle
from itertools import accumulate
from operator import add

import scipy.stats

from matplotlib import pyplot as plt
import seaborn as sns

from IPython.core.display import HTML


def formatDataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))


def groupData(data,measure,difficulties):
    groupedData = [data[measure,difficulty]
                        for difficulty in difficulties]
    return groupedData

def estimate_f_distribution(groupedData,N=10000):
    fs = [scipy.stats.f_oneway(*generatePermutedData(groupedData)).statistic
          for _ in range(N)]
    return fs

def generatePermutedData(listOfGroupedData):
    
    numGroups = len(listOfGroupedData)
    groupIndices = [0] + list(accumulate(  
                                            [len(group) 
                                            for group in listOfGroupedData],
                                         
                                         add)
                             )

    allData = [datapoint for group in listOfGroupedData for datapoint in group]
    
    shuffle(allData)

    groupPermutedData = [allData[groupIndices[ii]:groupIndices[ii+1]]
                                for ii in range(numGroups)]
    
    return groupPermutedData

def plotApproximatedF(fs):
    hist_style = {'alpha': 0.7,
            'edgecolor': 'gray',
            'linewidth': 4,
            'histtype': 'stepfilled',
            'normed':True}

    sns.distplot(fs,kde=False,hist_kws=hist_style);
    plt.title("Approximate Re-Sampling F-Distribution");
    return

def simulateNull(groupedData,N=1000):
    ps = [scipy.stats.f_oneway(*generatePermutedData(groupedData)).pvalue
             for _ in range(N)]
    return ps