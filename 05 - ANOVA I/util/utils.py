from random import shuffle
from itertools import accumulate
from operator import add

import scipy.stats

from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np

def group_data(data,measure,difficulties):
    grouped_data = [data[measure,difficulty]
                        for difficulty in difficulties]
    return grouped_data

def estimate_F_distribution(grouped_data,N=10000):
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

    group_permuted_data = [all_data[group_indices[ii]:group_indices[ii+1]]
                                for ii in range(num_groups)]

    return group_permuted_data

def plot_approximated_F(fs):
    hist_style = {'alpha': 0.7,
            'edgecolor': 'gray',
            'linewidth': 4,
            'histtype': 'stepfilled',
            'normed':True}

    sns.distplot(fs, kde=False, hist_kws=hist_style,
                        label=r'Resampling-Estimated PDF of $F$-statistic');
    plt.title("Null Distribution of F Statistic")
    plt.legend()
    return

def plot_true_F(dof_1, dof_2, F_min = 0.001, F_max = 10, n_steps = 100):
    Fs = np.linspace(F_min, F_max, num=n_steps)
    true_pdf = [scipy.stats.f.pdf(F, dof_1, dof_2) for F in Fs]
    plt.plot(Fs,true_pdf,
             color = 'r', linewidth=4,
             label = r"True PDF of $F$-statistic");
    plt.legend()

def simulate_null(grouped_data,N=1000):
    ps = [scipy.stats.f_oneway(*generate_permuted_data(grouped_data)).pvalue
             for _ in range(N)]
    return ps

def anova_by_hand(dataframe, measure):

    N = len(dataframe[measure])
    groups = dataframe["difficulty"].unique()
    k = len(groups)
    group_size = N/k

    anova_frame = make_anova_frame(dataframe, measure, groups)

    sum_of_squares, dof, mean_square = make_anova_dicts(anova_frame, measure, N, k)

    F = mean_square["explained"]/mean_square["residual"]

    return anova_frame, sum_of_squares, dof, mean_square, F

def make_anova_frame(dataframe, measure, groups):
    anova_frame = dataframe.copy()
    anova_frame["grand_mean"] = anova_frame[measure].mean()

    group_means = anova_frame.groupby("difficulty")[measure].mean()

    for group in groups:
        anova_frame.loc[anova_frame.difficulty==group,"group_mean"] = group_means[group]

    anova_frame["explained"] = anova_frame["group_mean"]-anova_frame["grand_mean"]

    anova_frame["residual"] = anova_frame[measure]-anova_frame["group_mean"]

    return anova_frame

def make_anova_dicts(anova_frame, measure, N, k):

    sum_of_squares = {}

    keys = [measure, "grand_mean", "explained", "residual"]

    for key in keys:
        sum_of_squares[key] = np.sum(np.square(anova_frame[key]))

    sum_of_squares["explainable"] = sum_of_squares[measure] - sum_of_squares["grand_mean"]

    dof = {}
    dof_vals = [N, 1, k-1, N-1]

    for key, dof_val in zip(keys, dof_vals):
        dof[key] = dof_val

    mean_square = {}

    for key in ["explained", "residual"]:
        mean_square[key] = sum_of_squares[key]/dof[key]

    return sum_of_squares, dof, mean_square
