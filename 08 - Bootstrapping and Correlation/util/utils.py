import numpy as np
import scipy.stats

import matplotlib.pyplot as plt

import time

def plot_bootstrap(sampler = np.random.standard_normal, pdf = scipy.stats.norm(loc=0, scale=1).pdf,
                   statistic_func = np.mean, num_samples=100,
                   animate_sampling=True, plot_statistic=False,
                  num_bootstraps=10,
                  num_alternates=10,):

    data = sampler(size=num_samples)
    statistic = statistic_func(data)

    bootstraps = [np.random.choice(data,size=num_samples) for _ in range(num_bootstraps)]
    bootstrap_statistics = [statistic_func(bootstrap) for bootstrap in bootstraps]

    alternate_samples = [sampler(size=num_samples) for _ in range(num_alternates)]
    alternate_statistics = [statistic_func(alternate_sample) for alternate_sample in alternate_samples]

    if animate_sampling:

        sns.distplot(data, kde=False,
                     hist_kws={'histtype':'step', 'linewidth':6, 'alpha':0.8, 'normed':True, 'zorder':6},
                     label = 'original data sample');

        [sns.distplot(alternate_sample, kde=False, color = 'chartreuse',
                    hist_kws={'histtype':'step', 'linewidth':6, 'alpha':0.25, 'normed':True},
                    label = 'other data samples')
                    for alternate_sample in alternate_samples];

        [sns.distplot(bootstrap, kde=False, color = 'hotpink',
                     hist_kws={'histtype':'step', 'linewidth':6, 'alpha':0.25, 'normed':True},
                     label = 'bootstrap samples')
                     for bootstrap in bootstraps];

        xs = np.linspace(-4,4,1000)
        plt.plot(xs,pdf(xs), color='black', linewidth=6, zorder=0,
                label = 'population distribution');

        handles, labels = plt.gca().get_legend_handles_labels()
        indices = [0, 1, 1+num_bootstraps, 1+num_bootstraps+num_alternates]

        handles = [handles[index] for index in indices]
        labels = [labels[index] for index in indices]

        plt.legend(handles, labels, loc=(1,0.6));

    elif plot_statistic:
        if statistic_func is not np.mean:
            raise NotImplementedError("statistic_func must be np.mean")

        sns.distplot(alternate_statistics, kde=False, color = 'chartreuse',rug=True,
            hist_kws={'histtype':'step', 'linewidth':6, 'alpha':0.5, 'normed':True},
            label = 'other data sample statistics')

        sns.distplot(bootstrap_statistics, kde=False, color = 'hotpink',rug=True,
                     hist_kws={'histtype':'step', 'linewidth':6, 'alpha':0.5, 'normed':True},
                     label = 'bootstrap sample statistics')

        scaling_factor = 1/np.sqrt(num_samples)
        xs = np.linspace(-4*scaling_factor, 4*scaling_factor, 1000)

        plt.plot(xs,scipy.stats.norm(loc=0, scale=scaling_factor).pdf(xs), color='black', linewidth=6, zorder=0,
                label = 'sampling distribution');

        ylims = plt.gca().get_ylim()
        plt.vlines(statistic, *ylims,
                   colors='grey', linestyles='--', linewidth=6,
                  label = 'value of statistic on data')
        plt.gca().set_ylim(ylims)

        plt.legend(loc=(1,0.6))

    return

def make_eCDF(data):

    def eCDF(x):
        N = len(data)
        total_below = data <= x
        accumulated_probability = np.sum(total_below)/N
        return accumulated_probability

    return eCDF

def animate_samples(sampler=np.random.standard_normal,
               true_CDF=scipy.stats.norm.cdf,
               input_range = np.arange(-5,5,0.1),
               sample_size = 50,
               num_samples = 10):
    """
    produces an animated plot of num_samples samples
    of size sample_size from sampler
    and compares them to the true_CDF

    CDFs for many of the samplers implemented in numpy.random
    are available via scipy.stats as .cdf methods
    of the matched random variable class.
    Default arguments provide an example,
    the normal distribution.
    """


    if num_samples >= 25:
        alpha = 0.25
    else:
        alpha=0.4

    fig = plt.figure()

    plt.plot(input_range,true_CDF(input_range),
            linewidth=8,
             color='k',label='true CDF')

    plt.legend(loc='lower right')

    plt.xlabel('x',fontsize=24)
    plt.ylabel('P(X ≤ x)',fontsize=24)

    fig.canvas.draw()
    time.sleep(0.75)

    for idx in range(num_samples):

        samples = sampler(size=sample_size)

        eCDF = make_eCDF(samples)

        plt.plot(input_range,
                [eCDF(input) for input in input_range],
                linewidth=2,alpha=alpha,
                color = 'hotpink',label='single sample eCDF')

        if idx == 0:
            plt.legend(loc='lower right')

        fig.canvas.draw()

def animate_sampling(sampler=np.random.standard_normal,
                      true_CDF=scipy.stats.norm.cdf,
                      input_range = np.arange(-5,5,0.1),
                      sample_sizes = [1,2,3,5,
                                     10,20,30,50,
                                     100,200,300,500,
                                     1000]):

    fig = plt.figure()
    samples = np.asarray([])

    plt.plot(input_range,true_CDF(input_range),
            linewidth=8,
             color='k',label='true CDF')

    plt.legend(loc=4)

    plt.xlabel('x',fontsize=24)
    plt.ylabel('P(X ≤ x)',fontsize=24)
    last_plotted_line = False

    for idx,sample_size in enumerate(sample_sizes):
        fig.canvas.draw()
        time.sleep(0.75)

        if last_plotted_line:
            last_plotted_line.set_visible(False)

        current_size = len(samples)

        new_samples = sampler(size=sample_size-current_size)
        samples = sorted(np.hstack([samples,new_samples]))
        eCDF = make_eCDF(samples)

        line_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

        last_plotted_line, = plt.plot(input_range,
                                     [eCDF(input) for input in input_range],
                                    linewidth=4, alpha=.8,
                                    color=line_color,
                                   label = 'sample eCDF')
        plt.title('Sample Size = '+ str(sample_size))

        if idx == 0:
            plt.legend(loc=4)

def animate_bootstraps(sampler=np.random.standard_normal,
                    true_CDF=scipy.stats.norm.cdf,
                    input_range = np.arange(-5,5,0.1),
                    sample_size = 50,
                    num_bootstraps = 10) :

    if num_bootstraps>25:
        alpha = 0.25
    else:
        alpha=0.4

    fig = plt.figure()
    samples = sorted(sampler(size=sample_size))

    eCDF = make_eCDF(samples)

    plt.plot(input_range,true_CDF(input_range),
            linewidth=8,
             color='k',label='true CDF')

    plt.legend(loc='lower right')

    plt.xlabel('x',fontsize=24)
    plt.ylabel('P(X ≤ x)',fontsize=24)

    fig.canvas.draw()
    time.sleep(0.75)

    plt.plot(input_range,[eCDF(input) for input in input_range],
    linewidth=8,label='empirical CDF')

    plt.legend(loc='lower right')

    fig.canvas.draw()
    time.sleep(0.75)

    handles, labels = plt.gca().get_legend_handles_labels()

    for idx in range(num_bootstraps):

        boot_samples = np.random.choice(samples, size=len(samples))

        boot_CDF = make_eCDF(boot_samples)

        h, = plt.plot(input_range,
                [boot_CDF(input) for input in input_range],
                linewidth=1, alpha=alpha,
                color='hotpink', label='single bootstrap CDF')

        if idx == 0:
            handles.append(h)
            labels.append('single bootstrap CDF')
            plt.legend(handles, labels, loc='lower right')
        fig.canvas.draw()

    time.sleep(0.75)

    for idx in range(num_bootstraps):

        alternate_samples = sampler(size=sample_size)

        alternate_CDF = make_eCDF(alternate_samples)

        h, = plt.plot(input_range,
                [alternate_CDF(input) for input in input_range],
                linewidth=1,alpha=alpha,
                color = 'chartreuse',label='single sample CDF')

        if idx == 0:
            handles.append(h)
            labels.append('single sample CDF')
            plt.legend(handles, labels, loc='lower right')
        fig.canvas.draw()
