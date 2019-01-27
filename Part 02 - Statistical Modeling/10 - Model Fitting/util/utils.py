import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import scipy.stats

###
# 3D regression plot
###

def make_regression_plot(fit_best_model):

    #first make the data
    x_spread = 0.5
    noise_level = 0.5

    # regressors are
    x = np.asarray([
    [0.5,-1,0.5],
    [1,1,1]])

    # for aesthetic reasons, we want the xs to be normalized
    normalized_x = np.asarray([x[0,:]/np.sqrt(np.sum(np.square(x[0,:]))),
                         x[1,:]/np.sqrt(np.sum(np.square(x[1,:])))]
                        )

    y = 0.5*normalized_x[0]+np.random.standard_normal(3)*noise_level

    N = 5
    mesh = np.linspace(-1,
                   1,
                   N)
    weights1,weights2 = np.meshgrid(mesh,mesh)

    Xs = normalized_x[0,0]*weights1+normalized_x[1,0]*weights2
    Ys = normalized_x[0,1]*weights1+normalized_x[1,1]*weights2
    Zs = normalized_x[0,2]*weights1+normalized_x[1,2]*weights2

    # setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax._axis3don = False

    # plot the "achievable plane" of model outputs
    ax.plot_surface(Xs,Ys,Zs,alpha=0.5,color='hotpink',shade=False)

    # grab the limits of the original plot before we equalize the axis
    ax_lim = plt.xlim()

    # we'll use them to plot the axes:
    #   an axis is a line that goes between the limits on one axis,
    #    while staying at 0 on the other axes
    coords = [ax_lim, np.zeros_like(ax_lim), np.zeros_like(ax_lim)]

    #  plt's set aspect doesn't work in 3D, so use this homebrew
    axis_equal_3d(plt.gca())

    # plot the y, z, and x axes as thick black lines
    for _ in range(3):
        coords = np.roll(coords, 1, axis=0)
        plt.plot(*coords,
         color='k', linewidth=4, zorder=1);

    # compute the weights of the best-fit model
    if fit_best_model:
        weights = ordinary_least_squares(normalized_x,y)
    else:
        weights = np.asarray(np.random.rand(2)*x_spread)

    # compute and plot the outputs of the model as a black star
    y_hats = np.squeeze(np.dot(weights, normalized_x))
    ax.scatter3D(*y_hats, marker='*', s=24**2, edgecolor='k', facecolor='None',
                     zorder=0, zdir='z', linewidth=2)

    # connect the true values and the model outputs with a red line
    #   to represent the residuals
    #   marking the true values with a blue dot
    residual_line = zip(y,y_hats)
    ax.plot(*residual_line, linewidth=4, color='red', marker='.', markerfacecolor='skyblue',
            markersize=24, markevery=2);

    return


def ordinary_least_squares(regressors, targets):
    arrays = [regressors, targets]

    arrays = [np.expand_dims(array,0) if array.ndim == 1
              else array
              for array in arrays]

    regressors, targets = arrays

    inverse_gram_matrix = np.linalg.inv(np.dot(regressors,regressors.T))
    weights = np.dot(np.dot(targets, regressors.T), inverse_gram_matrix)
    return weights

def axis_equal_3d(ax,center=0):
    # FROM StackO/19933125

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    if center == 0:
        centers = [0,0,0]
    else:
        centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
###
# Computing and Plotting residuals and regressor-residual correlations
###

def run_simulation(num_experiments, num_datapoints, use_ground_truth, gaussian_noise, noise_level):

    true_W = 2

    rs = []

    for _ in range(num_experiments):

        xs = np.asarray(np.random.standard_normal(num_datapoints))

        if gaussian_noise:
            noise = np.random.standard_normal(num_datapoints)
        else:
            noise = np.multiply(np.random.choice([-1,1], size=num_datapoints),
                                np.random.standard_exponential(num_datapoints))

        noise = noise*noise_level

        ys = true_W*xs+noise

        if use_ground_truth:
            w = true_W
        else:
            w = ordinary_least_squares(xs,ys)

        residuals = ys-w*xs

        #all_residuals = np.hstack([all_residuals,np.squeeze(residuals)])
        rs.append(np.corrcoef(residuals,xs)[0,1])

    return xs, ys, rs, residuals

def plot_simulation_results(example_xs, example_ys, rs, example_residuals):

    plt.figure()
    ax = sns.distplot(rs, hist=True, kde=False, rug=False,
                 hist_kws={'histtype':'stepfilled','linewidth':4,'density':True});
    ax.set_xlim([-1,1])
    plt.title('Distribution of Correlations Between Residuals and Regressors')

    plt.figure()
    plt.scatter(example_xs,example_ys)
    plt.title('Example Dataset')


    plt.figure()
    sns.distplot(example_residuals,hist=True,kde=False,rug=False,
                 hist_kws={'histtype':'stepfilled','linewidth':4,'density':True})
    plt.title('Example Distribution of Residuals');

    return
###
# Plotting Cost Surfaces
###

from scipy.signal import convolve

def gauss_random_field(x,y,scale):
    white_field = np.random.standard_normal(size=x.shape)

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    gauss_rv = scipy.stats.multivariate_normal([0,0], cov=np.ones(2))
    gauss_pdf = gauss_rv.pdf(pos)
    red_field = scale*convolve(white_field, gauss_pdf, mode='same')
    return red_field

def plot_cost_surface(cost, N, mesh_extent):
    mesh = np.linspace(-mesh_extent, mesh_extent, N)
    weights1, weights2 = np.meshgrid(mesh, mesh)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax._axis3don = False

    ax.plot_surface(weights1, weights2, cost(weights1, weights2),
                rstride=2, cstride=2, linewidth=0.2, edgecolor='b',
                alpha=1, cmap='Blues', shade=True);

    axis_equal_3d(plt.gca(), center=True)
