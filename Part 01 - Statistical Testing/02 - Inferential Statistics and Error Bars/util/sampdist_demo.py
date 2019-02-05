import itertools

import matplotlib.pyplot as plt
import numpy as np


def calc_streaks(outcomes):
    """
    Calculate all of the streaks (sub-sequences of identical outcomes)
    in a list of outcomes.

    Parameters
    ----------
    outcomes : list, a sequence of outcomes, e.g. a data sample

    Returns
    -------
    streaks : list, a sequence of tuples describing the streaks in outcomes.
              each tuple has the outcome's label as the first element and
              the length of the streak as the second
    """
    streaks = []

    # supremely hacky way to compute streaks,
    # thanks to StackOverflow/28839607
    for label, group in itertools.groupby(outcomes):
        length = sum(1 for elem in group)
        streaks.append((label, length))

    return streaks


def plot_sampling_distribution(pmf, ax=None, title=None):

    xticklabels = []
    for key in pmf.keys():
        if hasattr(key, "__iter__"):
            xticklabels.append("".join(str(elem) for elem in key))
        else:
            xticklabels.append(str(key) if not isinstance(key, np.float)
                               else "{:.3f}".format(key))

    xticks = range(len(pmf))

    if ax is None:
        f, ax = plt.subplots()

    ax.bar(xticks, pmf.values(), width=1.0, linewidth=4, edgecolor="k")

    if max(len(xticklabel) for xticklabel in xticklabels) > 4:
        plt.setp(ax.get_xticklabels(),
                 rotation=30, horizontalalignment='right')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel(r"$\mathrm{Outcome}$", fontsize=20)
    ax.set_ylabel(r"$p(\mathrm{Outcome})$", fontsize=20)

    if title is not None:
        ax.set_title(title, fontsize=32)
