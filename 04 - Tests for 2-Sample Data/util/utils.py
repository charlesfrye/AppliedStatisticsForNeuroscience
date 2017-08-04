import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import HTML
import numpy as np

def formatDataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))

def differencePlot(tidyFrame,ylabel="outcome"):
    """
    Accepts as first arg a tidy-format dataFrame with
    exactly two numeric-valued columns and plots it as a
    "difference plot" -- side-by-side stripplots for each
    numeric column, with observations from the same row
    joined by a line.

    Optionally also accepts a label for the y-axis as kwarg ylabel.
    """

    sns.stripplot(data=tidyFrame,
                  size=16,linewidth=4,
                  edgecolor="k",color='lightcoral');

    # from StackOverflow 25039626 -- select only numeric columns
    numerics = tidyFrame.select_dtypes(include=[np.number])

    assert numerics.shape[1] == 2, "dataFrame needs exactly two numeric columns!"

    xs = np.asarray([[0,1]]*(numerics.shape[1])).T
    ys = np.asarray(numerics).T

    plt.plot(xs,ys,
             linewidth=4,color='k');

    plt.ylabel(ylabel)

def make_shuffle(tidy_dataframe,):
    """
    Makes and returns a shuffled copy of a tidy dataframe with two columns.
    Here, "shuffling" means swapping the column indices for a random subset
    of rows, as is needed for performing a randomization test
    on a test statistic computed from column differences in the original dataframe.
    """

    shuffle = tidy.copy()

    for index in range(tidy.shape[0]):
        if np.random.choice([True, False]):
            shuffle.iloc[index,:] = tidy.iloc[index,::-1].as_matrix()

    return shuffle
