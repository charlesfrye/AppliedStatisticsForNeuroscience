import time
import numpy as np
import matplotlib.pyplot as plt


def setup_run(pmf, iters):
    pmfs = [pmf]
    x_max = iters * (len(pmf))
    x_locations = list(range(x_max+2))
    x_labels = [str(loc) if (loc % (len(pmf) - 1)) == 0 else ''
                for loc in x_locations]
    extended_PMF = np.hstack([pmfs[0], [0] * (x_max + 2 - len(pmfs[0]))])
    edge = 2
    fig = plt.figure(figsize=(12, 6))
    pmf_ax = plt.subplot(111)
    pmf_bars = pmf_ax.bar(x_locations, extended_PMF,
                          width=1, align='center', alpha=0.8,
                          linewidth=0,)

    setup_plot(plt.gca(), x_locations, edge, x_labels)

    plt.suptitle("Adding Up "+str(iters)+" Random Numbers",
                 size=24, weight='bold', y=1.)
    fig.canvas.draw()

    return fig, pmf_bars, pmfs


def setup_plot(ax, locs, edge, labels):
    ax.set_ylim([0, 1])
    ax.set_xlim([locs[0] - edge, locs[1] + edge])
    ax.xaxis.set_ticks(locs)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticks([0, 0.5, 1])
    ax.tick_params(axis='x', top=False)
    ax.tick_params(axis='y', right=False)
    plt.ylabel('Probability', fontsize='x-large', fontweight='bold')


def central_limit_demo(pmf, iters):
    """
        Recursively convolves pmf with itself iters times
        and draws the results to a figure with delays.

        Recursive convolution gives the pmf for adding random
        variables from the same distribution,
        so the resulting pmfs are the distributions of
        sums of independent and identically distributed random variables
    """
    assert min(pmf) >= 0, "no negative numbers in pmf"
    assert np.isclose(sum(pmf), 1), "doesn't sum to 1"
    assert max(pmf) < 1, "must have non-zero variance"

    figure, bar_plot, pmfs = setup_run(pmf, iters)
    time.sleep(0.2)

    for _ in range(iters):
        [bar_plot[idx].set_height(h)
         for idx, h in enumerate(pmfs[-1])]
        pmfs.append(np.convolve(pmfs[-1], pmfs[0]))
        figure.canvas.draw()
        time.sleep(0.1 * (1 - 0.1) ** _)
