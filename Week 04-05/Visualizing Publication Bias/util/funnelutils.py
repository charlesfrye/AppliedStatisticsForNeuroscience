from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import distributions

class Experiment(object):
    def __init__(self, true_variance,random_indices,
                    ):
        super(Experiment, self).__init__()
        self.true_variance = true_variance
        self.random_indices = self.random_indices


# Helper functions to simulate experiments.
def simulate_data(effect, variance, n,experiment):
    """Simulate a population of data. We'll sample from this in each study.
    Note that we're drawing from a normal distribution."""
    data = np.sqrt(experiment.true_variance) * np.random.randn(int(n))
    data += effect
    return data

def simulate_experiments(data, experiment, n_min=10, n_max=50, prefer_low_n=False,
                         n_simulations=100):
    """Randomly simulates data collection and analyses of many experiments.
    
    On each iteration, it chooses a random sample from data, calculates the
    mean of that sample, as well as a p-value associated with that mean's
    difference from 0.
    
    data : the full population dataset
    n_min : the minimum sample size for each study.
    n_max : the maximum sample size for each study.
    prefer_low_n : whether lower sample sizes are preferred.
    """
    effects = np.zeros(n_simulations)
    n = np.zeros(n_simulations)
    p = np.zeros(n_simulations)
    for ii in range(n_simulations):
        # Take a random sample from the population
        if prefer_low_n is False:
            n_sample = np.random.randint(n_min, n_max, 1)[0]
        else:
            probabilities = np.logspace(5, 1, n_max - n_min)
            probabilities /= np.sum(probabilities)
            n_sample = np.random.choice(range(n_min, n_max),
                                        p=probabilities)
        ixs_sample = experiment.random_indices[ii][:n_sample]
        i_data = data[ixs_sample]
        effects[ii] = np.mean(i_data)
        n[ii] = n_sample
        p[ii] = calculate_stat(np.mean(i_data), np.std(i_data), n_sample)
    return effects, n, p

def calculate_stat(mean, std, n, h0=0):
    """Calculate a p-value using a t-test.
    
    Note that this probably *isn't* the right test to run with data that
    is bounded on either side (in this case, -1 and 1). However, luckily
    this is not a statistics tutorial so I'm just going to be blissfully
    ignorant of this.
    """
    t = (mean - h0) / (std / np.sqrt(n))
    p = distributions.t.pdf(t, n-1)
    return p


def plot_funnel_plot(effects, sample_sizes,
                     effects_reported, sample_sizes_reported,
                     p_effects_reported):
    """Creates a funnel plot using a 'full' set of effects, corresponding
    to the effects we'd report if all results were published, regardless of
    their 'significance', as well as a 'reported' set of effects which made
    it through peer review"""
    # Create a figure w/ 2 axes
    fig = plt.figure(figsize=(5, 5))
    axdist = plt.subplot2grid((4, 4), (0, 0), 1, 4)
    axmesh = plt.subplot2grid((4, 4), (1, 0), 3, 4)

    # Calculate relevant stats
    mn_full = effects.mean()
    std_full = effects.std()
    mn_pub = effects_reported.mean()
    std_pub = effects_reported.std()
    
    mn_diff = np.abs(mn_full - mn_pub)
    std_diff = np.abs(std_full - std_pub)
    
    # First axis is a histogram of the distribution for true/experimental effects
    bins = np.arange(-2, 2, .1)
    _ = axdist.hist(effects, color='k', histtype='stepfilled',
                    normed=True, bins=bins)
    _ = axdist.hlines(4.5, mn_full - std_full, mn_full + std_full,
                      color='.3', lw=2)
    _ = axdist.hist(effects_reported, color='r', histtype='step', lw=2,
                    normed=True, bins=bins)
    _ = axdist.hlines(4.0, mn_pub - std_pub, mn_pub + std_pub,
                      color='r', lw=2)
    axdist.set_ylim([0, 5])
    axdist.set_title('Distribution of effects\nError in mean: {:.3f}'
                     '\nError in std: {:.3f}'.format(mn_diff, std_diff))
    axdist.set_axis_off()

    # Now make the funnel plot
    sig = pvals < .05
    mesh = axmesh.contour(combinations[0], combinations[1], sig, cmap=plt.cm.Greys,
                          vmin=0, vmax=3, rasterized=True)
    
    inv_p_effects = 1 - p_effects_reported
    axmesh.scatter(effects, sample_sizes,
                   s=100, c='k', alpha=.1)
    axmesh.scatter(effects_reported, sample_sizes_reported,
                   s=100, c=inv_p_effects,
                   vmin=.95, vmax=1., cmap=plt.cm.viridis)
    axmesh.axis('tight')
    axmesh.set_xlabel('Effect Size')
    axmesh.set_ylabel('Sample Size (or statisical power)')

    _ = plt.setp(axdist, xlim=axmesh.get_xlim())
    return fig