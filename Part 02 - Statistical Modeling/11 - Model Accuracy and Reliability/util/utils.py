import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

def ordinary_least_squares(regressors, targets):
    arrays = [regressors, targets]

    arrays = [np.expand_dims(array,0) if array.ndim == 1
              else array
              for array in arrays]

    regressors, targets = arrays

    gram_matrix_inverse = np.linalg.inv(np.dot(regressors, regressors.T))
    weights = np.dot(np.dot(targets, regressors.T), gram_matrix_inverse)
    return weights

def SS(xs):
    return np.sum(np.square(xs))

def leave_out_from(xs, index):
    return np.hstack([xs[:,:index], xs[:,index+1:]])

def leave_one_out_cv(xs,ys):
    N = xs.shape[1]; ws = np.zeros_like(xs);
    denominator = SS(ys-np.mean(ys))

    for idx in range(N):
        ws[:,idx] = ordinary_least_squares(leave_out_from(xs,idx),
                                            leave_out_from(ys,idx))

    predictions = [np.dot(ws.T[idx], xs.T[idx]) for idx in range(N)]
    R_squared = 1 - SS(ys-predictions)/SS(ys-np.mean(ys))

    return R_squared

def compute_R_squared(xs,ys,w):
    residuals = ys-np.dot(w.T,xs)
    R_squared = 1 - SS(residuals)/SS(ys-np.mean(ys))
    return R_squared

def run_simulation(true_degree, population_size=1000, sample_size=10,
                 min_degree=0, max_degree=10, num_experiments=25, x_limits=[-1.5,1.5]):

    weights = [0.5, 1, -1.75, 0.7, -0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
    weights = np.atleast_2d(weights[:true_degree+1]).T

    plt.figure()
    plot_model(weights, limits=x_limits, label='True Model')

    base_xs = np.random.uniform(*x_limits, size=population_size)

    x_population = np.asarray([np.ones(population_size), base_xs])

    for degree in range(2,true_degree+1):
        x_population = np.vstack([x_population, np.power(x_population[None, 1, :], degree)])

    noise_level = 0.75
    y_population = np.dot(weights.T, x_population) + np.random.standard_normal(population_size)*noise_level

    R_squared_CV = np.zeros((max_degree + 1 - min_degree, num_experiments))
    R_squared_fitted = np.zeros((max_degree + 1 - min_degree, num_experiments))
    R_squared_actual = np.zeros((max_degree + 1 - min_degree, num_experiments))

    plt.scatter(base_xs.T, y_population.T, alpha=0.1, s=72)

    for experiment in range(num_experiments):
        model_xs = x_population[None, 0, :]

        indices = np.random.choice(population_size, size=sample_size)

        y_sample = y_population[:, indices]

        if experiment == 0:
            plt.scatter(np.squeeze(x_population[1, indices]),
                    np.squeeze(y_sample),
                     alpha=1, s=48, color='k', zorder=20)

        for degree in range(min_degree, max_degree+1):
            if degree > 0:
                model_xs = np.vstack([model_xs, np.power(x_population[None, 1, :], degree)])

            degree_index = degree - min_degree
            x_sample = model_xs[:, indices]

            R_squared_CV[degree_index,experiment] = leave_one_out_cv(x_sample, y_sample)

            w_fitted = ordinary_least_squares(x_sample,y_sample).T

            if experiment == 0:

                if degree in [0, 1, 2, 3, 8, true_degree, max_degree]:
                    plot_model(w_fitted, limits=x_limits)

            R_squared_fitted[degree_index, experiment] = compute_R_squared(x_sample, y_sample, w_fitted)
            R_squared_actual[degree_index, experiment] = compute_R_squared(model_xs, y_population, w_fitted)

    plt.ylim([-10,10])
    plt.legend(loc='best')
    R_squared_actual = np.mean(R_squared_actual, axis=1)
    R_squared_CV = np.mean(R_squared_CV, axis=1)
    R_squared_fitted = np.mean(R_squared_fitted, axis=1)

    if max_degree >= 8:
        for idx in range(-1, max_degree - 10):
            if R_squared_fitted[idx] < 0.8:
                R_squared_fitted[idx] = 1

    return R_squared_fitted, R_squared_actual, R_squared_CV

def plot_model(weights, label=None, limits=[-2, 2]):
    N = 1000
    xs = np.linspace(*limits, num=N)
    input_xs = np.asarray([np.ones(N)])

    for degree in range(1, weights.shape[0]):
        input_xs =  np.vstack([input_xs, np.power(xs, degree)])

    if label == None:
        degree = weights.shape[0]-1
        label = str(degree)
        width = 2
        zorder = 15 - degree
    else:
        width = 6
        zorder= 1
    outputs = np.dot(weights.T, input_xs)
    plt.plot(xs, np.squeeze(outputs),
             linewidth=width, label=label, zorder=zorder)

def plot_results(fitted, actual, cross_validated, min_degree=0, max_degree=10):
    degrees = range(min_degree, max_degree + 1)

    plt.figure()
    for label, r_squared_estimate in zip(['sample','population','CV'],
                                  [fitted, actual, cross_validated]):
        plot_R_squared(degrees, r_squared_estimate, label)

    current_ymin = plt.gca().get_ylim()[0]
    plt.xlim(-0.5, max_degree + 0.5)
    plt.ylim(max(-0.2, current_ymin), 1.1)

    plot_axes()

    plt.ylabel('R**2'); plt.xlabel('Modeling Polynomial Degree')
    plt.legend(loc='best');

def plot_R_squared(degrees, values, label):
    plt.plot(degrees, values,
         linestyle='-', linewidth=6,
         marker='.', markersize=36, label=label)

def plot_axes():
    x_lim = plt.xlim()
    y_lim = plt.ylim()
    plt.hlines([0, 0], 0, x_lim[1], color='k', linewidth=4);
    plt.vlines([0, 0], 0, y_lim[1], color='k', linewidth=4)

def clean_lmplot():
    ax = plt.gca()
    ax.set_ylim(-15,15)

def plot_true_model(w,b):
    ax = plt.gca()
    x_lims = ax.get_xlim()
    mesh = np.linspace(*x_lims)
    plt.plot(mesh, w*mesh + b,
             color='k', linewidth=4,
             label='True Model')
    plt.legend(loc='best')

def setup_linear_model(N, gaussian_noise=True, slope=2, offset=0):

    noise_level = 2

    xs = np.random.normal(size=N)*3

    if gaussian_noise:
        noise = np.random.standard_normal(size=N)*noise_level
    else:
        noise = np.random.standard_cauchy(size=N)

    ys = slope*xs + offset + noise
    df = pd.DataFrame.from_dict({'x':xs, 'y':ys})

    return df
