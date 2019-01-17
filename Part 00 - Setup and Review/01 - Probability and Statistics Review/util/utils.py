import matplotlib.pyplot as plt
import scipy.integrate as integrate
from IPython.core.display import HTML
import numpy as np

def format_dataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))

def plot_PMF(PMF,title=''):
    plt.figure()
    if PMF == []:
        return
    assert all([p >= 0 for p in PMF]), "not a valid PMF: negative numbers"
    assert abs(1-sum(PMF))<=10**-5, "not a valid PMF: doesn't sum to 1"
    linewidth = 4
    num_elements = len(PMF)
    plt.plot(PMF,marker='o',linestyle='none',
         markerfacecolor='white',markeredgecolor='black',
         markeredgewidth=linewidth,markersize=12)
    plt.vlines(range(num_elements),[0]*num_elements,PMF,
         linewidth=linewidth)
    heighten_YLim(plt.gca())
    plt.title(title,fontsize=24)

def plot_PDF(PDF,title=''):
    plt.figure()
    eps = 10**-3; edge=10
    xs = np.arange(-edge*eps,1+edge*eps,eps)
    plt.plot(xs,[PDF(x) for x in xs],
            marker='None',linewidth=4)
    heighten_YLim(plt.gca())
    plt.title(title,fontsize=24)

def heighten_YLim(ax):
    ylim = ax.get_ylim()
    ylim = (ylim[0],ylim[1]*1.1)
    ax.set_ylim(ylim)

def integrates_to_one(PDF):
    integral, err_bound = integrate.quad(PDF,0,1)
    return abs(1-integral) <= err_bound
