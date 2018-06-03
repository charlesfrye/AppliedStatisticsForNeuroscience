import numpy as np

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def logistic(x):
    return 1/(1+np.power(math.e,-x))

def plot_logistic():
    xs = np.linspace(-10, 10, 50)
    plt.figure()
    plt.plot(xs, logistic(xs), linewidth=4, color='k');

def plot_linear_model(weights=[], N=20, extent=1):

    regressors = setup_data(extent, N)

    if len(weights) == 0:
        weights = np.random.random(size=(len(regressors) - 1, 1))*2 + 1
        weights = np.append(weights, 0)

    phi = lambda regressors: np.dot(weights.T, regressors)

    ys = compute_ys(phi, regressors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_data(ax, regressors, ys)

    mesh = np.linspace(-extent, extent, 25)
    x_mesh, y_mesh = np.meshgrid(mesh, mesh)

    output = logistic(weights[0]*x_mesh + weights[1]*y_mesh+weights[2:])

    plot_surface(ax, x_mesh, y_mesh, output)

    return



def plot_linearized_model(phi_f, N=100, extent=2):

    regressors = setup_data(extent, N)

    ys = compute_ys(phi_f, regressors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_data(ax, regressors, ys)

    mesh = np.linspace(-extent, extent, 25)
    x_mesh,y_mesh = np.meshgrid(mesh, mesh)

    output = logistic(phi_f([x_mesh, y_mesh]))

    plot_surface(ax, x_mesh, y_mesh, output)

    return

def plot_surface(ax, x_mesh, y_mesh, output):
    ax.plot_surface(x_mesh, y_mesh, output,
                rstride=2, cstride=2,
                shade=False, alpha=0.5, cmap='RdBu', zorder=1);
    return

def plot_data(ax, regressors, ys):
    ax.scatter(regressors[0], regressors[1], ys, depthshade=False, c=np.squeeze(ys),
                s=48, zorder=2, cmap='RdBu');
    return

def setup_data(extent,N):

    x1 = np.random.uniform(-extent, extent, size=(1, N))
    x2 = np.random.uniform(-extent, extent, size=(1, N))
    bias = np.ones_like(x1)

    regressors = np.vstack([x1 ,x2, bias])

    return regressors

def compute_ys(phi, Xs):
    return [np.random.binomial(1, logistic(phi(X))) for X in Xs.T]

def generate_linearized_models():

    quad_weights = [1, 1, -2]
    quad = lambda x: quad_weights[0]*np.square(x[0]) \
                          + quad_weights[1]*np.square(x[1])\
                            + quad_weights[-1]

    trig_weights = [4, 1, -1]
    trig = lambda x: 2*np.cos(trig_weights[0]*x[1]) \
                        + trig_weights[1]*(np.square(x[0]) + np.square(x[1]))\
                            + trig_weights[-1]

    check_weights = [1.5, 1.5, 1, 0]
    check = lambda x: np.where(x[0]*x[1]<0, check_weights[0]*x[0]*x[1],2) \
                        + np.where(x[0]*x[1]>0, check_weights[1]*x[0]*x[1],-2) \
                        + check_weights[2]*(np.square(x[0])+np.square(x[1]))\
                            + check_weights[-1]

    return {'quadratic':quad,
            'trig':trig,
            'checker':check}

#### Code below is from scikit-learn documentation
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# Modified for educational use by Charles Frye
# License: BSD 3 clause

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

def run_classifiers(classifiers, holdout=.75, N=100):

    names = ["Linear Classifier",
         "Neural Net",
         "Nearest Neighbors",
         ]

    h = .02  # step size in the mesh

    datasets = make_datasets(N)

    figure = plt.figure(figsize=(((len(classifiers) + 1)*2.5), len(datasets)*2.5 ))

    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=holdout, random_state=42)

        # make mesh
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        if ds_cnt == 0:
            ax.set_title("Input data")

        plot_train_and_test(X_train, y_train, X_test, y_test, holdout, ax)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            plot_train_and_test(X_train, y_train, X_test, y_test, holdout, ax)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')

            i += 1

    plt.tight_layout()
    plt.show()

def make_datasets(N=100):
    X, y = make_classification(n_samples=N, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    return [linearly_separable,
            make_moons(n_samples=N, noise=0.3, random_state=4),
            make_circles(n_samples=N, noise=0.2, factor=0.5, random_state=3),
            ]

def plot_train_and_test(train, train_labels, test, test_labels, holdout, ax):
    ax.scatter(train[:, 0], train[:, 1], c=train_labels, s=36, cmap='RdBu')
    ax.scatter(test[:, 0], test[:, 1], c=test_labels, alpha=0.35, s=36, cmap='RdBu')
    return
