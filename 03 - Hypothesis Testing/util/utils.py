import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import HTML
import numpy as np

def format_dataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))

def plot_rates(critical_values, rates, rate_type):
    plt.figure()
    
    plt.plot(critical_values,rates,
         linewidth=4,
         label=rate_type);
    
    plt.title(rate_type + " as a Function of Critical Value",
         fontsize='x-large',
          fontweight='bold');

    plt.xlabel('Critical Value'); plt.ylabel("Estimated Rate")
    plt.legend();
    
def plot_true_and_false_positive_rates(critical_values, false_positive_rates, true_positive_rates):

    plt.figure()
    plt.plot(critical_values,false_positive_rates,
             linewidth=4,
             label='False Positive Rate');
    
    plt.plot(critical_values,true_positive_rates,
             linewidth=4,
             label='True Positive Rate');
    
    plt.title("True and False Positive Rate as a Function of Critical Value",
             fontsize='x-large',
              fontweight='bold',
             y=1.02);
    
    plt.xlabel('Critical Value'); plt.ylabel("Estimated Rate")
    plt.legend();
    
def estimate_TPR(baserate, effect_sizes, critical_values, simulator,
                 num_experiments=10000):
    
    num_effect_sizes = len(effect_sizes)
    num_critical_values = len(critical_values)
    
    TPR_mat = np.zeros((num_effect_sizes,num_critical_values))
    
    for rowIdx,effect_size in enumerate(effect_sizes):
        for colIdx,critical_value in enumerate(critical_values):
            TPR_mat[rowIdx, colIdx] = np.mean(simulator(num_experiments,
                                                         baserate, effect_size,
                                                           critical_value))
    
    return TPR_mat

def plot_TPR(critical_values, effect_sizes, TPR):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    critical_value_mesh, effect_size_mesh = np.meshgrid(critical_values,effect_sizes)

    ax.plot_surface(effect_size_mesh,critical_value_mesh,TPR,
                    rstride=1,cstride=1,
                   cmap='RdBu',alpha=0.5);
    ax.set_xlabel('Effect Size'); ax.set_ylabel('Critical Value'); ax.set_zlabel('True Positive Rate');
    
def plot_null_and_result(null_values, result_value):
    
    bins = get_bins(null_values, result_value)
    
    plt.figure(); ax = plt.subplot(111)
    
    plot_null_distribution(ax, null_values, bins)
    
    plot_experimental_result(ax, result_value)
    
    plt.xlabel("Number of Spikes Observed")
    plt.ylabel("Probability")
    plt.legend()
    
    return

def get_bins(null_values, result_value):
    
    x_max = max(max(null_values), result_value)
    bins = np.arange(-.5,x_max,1)
    
    return bins

def plot_null_distribution(ax, null_values, bins):
    
    plt.sca(ax)
    
    sns.distplot(null_values,
             bins=bins,
             kde=False,hist=True,
             hist_kws={'normed':True,
                       'linewidth':4,
                       'edgecolor':'w',
                       'histtype':'stepfilled'},
             label='Null Distribution'
            );
    
    return

def plot_experimental_result(ax, result_value):
    
    plt.sca(ax)
    
    plt.vlines(result_value,
           ymin=0,
           ymax=ax.get_ylim()[1]/20,
          label='Observed Value');
    
    return


def make_CDF(probabilities, bin_edges):
    
    first_edge = bin_edges[0];
    leftpad_edges = np.arange(first_edge-10,first_edge,step=1)
    last_edge = bin_edges[-1]
    rightpad_edges = np.arange(last_edge+1,last_edge+11,step=1)
    new_edges = np.hstack([leftpad_edges,
                           bin_edges,
                           rightpad_edges ])
    
    probabilities = np.hstack([np.zeros(len(leftpad_edges)),
                               probabilities,
                               np.zeros(len(rightpad_edges))])
    
    CDF = lambda x: np.sum(probabilities[:np.argmax(new_edges>=x)])
    
    return CDF, new_edges

def plot_CDF(CDF, bin_edges):
    
    plt.figure()
    plt.plot(bin_edges, [CDF(bin_edge) for bin_edge in bin_edges],
             linewidth=4)
    
    plt.xlabel("Number of Spikes Observed")
    plt.ylabel("Cumulative Probability");
    plt.title("Cumulative Distribution Function",
             fontsize='x-large',
              fontweight='bold');
    
    return

def plot_p_distribution(ps):
    
    plt.figure()
    sns.distplot(ps,
                 bins=5,
                 kde=False,hist=True,
                 hist_kws={'normed':True,
                           'linewidth':4,
                           'edgecolor':'w',
                           'histtype':'stepfilled'},
                 label=r'Distribution of $p$-values'
                );

    _, y_max = plt.gca().get_ylim()
    plt.gca().set_ylim([0, y_max*2]);
    y_min, y_max = plt.gca().get_ylim()
    delta_y = y_max-y_min
    plt.gca().set_xlim([0.5-delta_y/2,0.5+delta_y/2])
    plt.gca().set_aspect('equal'); plt.legend();
    return