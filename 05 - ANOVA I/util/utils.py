from random import shuffle
from itertools import accumulate
from operator import add

import scipy.stats

from matplotlib import pyplot as plt
import seaborn as sns

def group_data(data,measure,difficulties):
    grouped_data = [data[measure,difficulty]
                        for difficulty in difficulties]
    return grouped_data

def estimate_f_distribution(grouped_data,N=10000):
    fs = [scipy.stats.f_oneway(*generate_permuted_data(grouped_data)).statistic
          for _ in range(N)]
    return fs

def generate_permuted_data(list_of_grouped_data):
    
    num_groups = len(list_of_grouped_data)
    group_indices = [0] + list(accumulate(  
                                            [len(group) 
                                            for group in list_of_grouped_data],
                                         
                                         add)
                             )

    all_data = [datapoint for group in list_of_grouped_data for datapoint in group]
    
    shuffle(all_data)

    group_permuted_data = [all_data[group_indices[ii]:groupIndices[ii+1]]
                                for ii in range(num_groups)]
    
    return group_permuted_data

def plot_approximated_F(fs):
    hist_style = {'alpha': 0.7,
            'edgecolor': 'gray',
            'linewidth': 4,
            'histtype': 'stepfilled',
            'normed':True}

    sns.distplot(fs,kde=False,hist_kws=hist_style);
    plt.title("Approximate Re-Sampling F-Distribution");
    return

def simulate_null(grouped_data,N=1000):
    ps = [scipy.stats.f_oneway(*generate_permuted_data(grouped_data)).pvalue
             for _ in range(N)]
    return ps
