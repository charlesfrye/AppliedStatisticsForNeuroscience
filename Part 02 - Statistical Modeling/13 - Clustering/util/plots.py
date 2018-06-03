import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from util.em import pull_cluster

color_list = ['red','darkcyan','brown','hotpink','plum','darkolivegreen',
             'skyblue','mediumspringgreen','indigo']

def plot_data(data, title):
    fig, axis = plt.subplots(1)
    axis.plot(data[0,:], data[1,:], '.',  c='gray', alpha=0.2, markersize=10)
    fig.suptitle(title)
    fig.show()

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def plot_likelihood(data, log_likelihoods, means, sigma, num_gauss, title):
    data_mean = np.mean(data) # MLE is the mean of the data
    z = (2*np.pi*sigma**2)**(1/2)
    best_gauss = 1/z * np.exp(-(data-data_mean)**2/(2*sigma**2))
    log_likelihood_gauss = np.mean(np.log(best_gauss))
    fig, ax = plt.subplots(2,figsize=(12,6))
    ax[0].plot(data_mean, log_likelihood_gauss, "*", color="k",
      markersize=10)
    ax[0].plot(means, log_likelihoods, "r")
    ax[0].set_ylabel("Log Likelihood", fontsize=12)
    ax[1].hist(data, normed=True,histtype='stepfilled',alpha=0.2,color='gray',bins=20)
    ymin,ymax = ax[0].get_ylim()
    xmin,xmax = ax[0].get_xlim()
    ax[0].vlines(0,ymin,ymax,linewidth=2,linestyle='dashed')
    gauss_data = np.linspace(np.min(data)-10, np.max(data)+10, 100)
    step = int(len(means)/num_gauss)
    gauss_idx_list = np.arange(0, len(means), step)
    ax[1].set_title(str(len(gauss_idx_list)) +
            " Example Gaussians and Data Distribution",fontsize=12)
    for gauss_idx in gauss_idx_list:
        z = (2*np.pi*sigma**2)**(1/2)
        gaussian = (1/z *
          np.exp(-(gauss_data-means[int(np.floor(gauss_idx))])**2 / (2 * sigma**2)))
        ax[1].plot(gauss_data, gaussian, linewidth=2)

    ax[1].set_xlim(xmin,xmax)
    fig.suptitle(title, fontsize=14)
    return fig

def plot_contour(mu,sigma,ax,color='blue',num_contours=3):
    eigvalues,eigvectors = np.linalg.eig(sigma)
    primary_eigvector = eigvectors[:,0]
    angle = compute_rotation(primary_eigvector)
    isoprob_contours = [Ellipse(mu,
                               l*np.sqrt(eigvalues[0]),
                               l*np.sqrt(eigvalues[1]),
                               alpha=0.3/l,color=color,
                              angle=angle)
                       for l in range(1,num_contours+1)]
    [plt.gca().add_patch(isoprob_contour) for isoprob_contour in isoprob_contours]

def compute_rotation(vector):
    return (180/np.pi)*np.arctan2(vector[1],vector[0])

def data_scatter(data,color='grey'):
    plt.scatter(data[:,0],data[:,1],color=color,edgecolor=None,alpha=0.1)
    return

def plot_observed_mix(dataset,means,sigmas,contours=True):
    plt.figure(figsize=(4,4))
    #color_list = ['darkcyan','hotpink','plum']
    min_idx = int(np.min(dataset[:,2]))
    max_idx = int(np.max(dataset[:,2]))
    mixture_elements = [pull_cluster(dataset,idx) for idx in range(min_idx,max_idx+1)]
    for idx,element in enumerate(mixture_elements):
        color_idx = idx % len(color_list)
        if not contours:
            data_scatter(element,color=color_list[color_idx])
        if contours:
            data_scatter(element)
            plot_contour(means[idx],sigmas[idx],plt.gca(),color=color_list[color_idx])
    return

def plot_unobserved_mix(dataset):
    plt.figure(figsize=(4,4))
    min_idx = int(np.min(dataset[:,2]))
    max_idx = int(np.max(dataset[:,2]))
    mixture_elements = [pull_cluster(dataset,idx) for idx in range(min_idx,max_idx+1)]
    for idx,element in enumerate(mixture_elements):
        data_scatter(element)
    return

def plot_EM_results(dataset,K,parameters):
    plt.figure()
    data_scatter(dataset)
    for idx in range(K):
        mu = parameters['mu'][idx]; sigma = parameters['sigma'][idx]
        color_idx = idx % (len(color_list))
        plot_contour(mu,sigma,plt.gca(),color=color_list[color_idx],num_contours=5)
