import matplotlib.pyplot as plt

def cleanPlot(axis,figure=None):
    """Removes spines and ticks from top and right sides of plot"""
    [axis.spines[side].set_visible(False) for side in ["top","right"]];
    axis.xaxis.set_ticks_position("bottom")
    axis.yaxis.set_ticks_position("left")
    return axis

def legendOutside(axis):
    """Places legend outside and to the right of a plot.
    Useful for times when loc="best" doesn't quite cut it."""
    axis.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
    return axis
