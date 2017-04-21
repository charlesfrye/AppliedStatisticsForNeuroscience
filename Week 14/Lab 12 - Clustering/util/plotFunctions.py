import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
import numpy as np
from util.util import pullCluster
colorList = ['red','darkcyan','brown','hotpink','plum','darkolivegreen',
             'skyblue','mediumspringgreen','indigo']

def plotData(data, title):
  fig, axis = plt.subplots(1)
  axis.plot(data[0,:], data[1,:], '.',  c='gray', alpha=0.2, markersize=10)
  fig.suptitle(title)
  fig.show()

def eigsorted(cov):
  vals, vecs = np.linalg.eigh(cov)
  order = vals.argsort()[::-1]
  return vals[order], vecs[:,order]

def plotLikelihood(data, logLikelihoods, means, sigma, numGauss, title):
  data_mean = np.mean(data) # MLE is the mean of the data
  z = (2*np.pi*sigma**2)**(1/2)
  bestGauss = 1/z * np.exp(-(data-data_mean)**2/(2*sigma**2))
  logLikelihoodGauss = np.mean(np.log(bestGauss))
  fig, ax = plt.subplots(2,figsize=(12,6))
  ax[0].plot(data_mean, logLikelihoodGauss, "*", color="k",
    markersize=10)
  ax[0].plot(means, logLikelihoods, "r")
  ax[0].set_ylabel("Log Likelihood", fontsize=12)
  ax[1].hist(data, normed=True,histtype='stepfilled',alpha=0.2,color='gray',bins=20)
  ymin,ymax = ax[0].get_ylim()
  xmin,xmax = ax[0].get_xlim()
  ax[0].vlines(0,ymin,ymax,linewidth=2,linestyle='dashed')
  gaussData = np.linspace(np.min(data)-10, np.max(data)+10, 100)
  step = int(len(means)/numGauss)
  gaussIdxList = np.arange(0, len(means), step)
  ax[1].set_title(
    str(len(gaussIdxList))+" Example Gaussians and Data Distribution",
    fontsize=12)
  for gaussIdx in gaussIdxList:
      z = (2*np.pi*sigma**2)**(1/2)
      gaussian = (1/z *
        np.exp(-(gaussData-means[int(np.floor(gaussIdx))])**2 / (2 * sigma**2)))
      ax[1].plot(gaussData, gaussian,linewidth=2)
  #ax[1].set_xlim(np.min(data), np.max(data))
  ax[1].set_xlim(xmin,xmax)
  fig.suptitle(title, fontsize=14)
  return fig

def plotContour(mu,sigma,ax,color='blue',numContours=3):
    eigvalues,eigvectors = np.linalg.eig(sigma)
    primaryEigvector = eigvectors[:,0]
    angle = computeRotation(primaryEigvector)
    isoProbContours = [Ellipse(mu,
                               l*np.sqrt(eigvalues[0]),
                               l*np.sqrt(eigvalues[1]),
                               alpha=0.3/l,color=color,
                              angle=angle) 
                       for l in range(1,numContours+1)]
    [plt.gca().add_patch(isoProbContour) for isoProbContour in isoProbContours]
    
def computeRotation(vector):
    return (180/np.pi)*np.arctan2(vector[1],vector[0])
    
def dataScatter(data,color='grey'):
    plt.scatter(data[:,0],data[:,1],color=color,edgecolor=None,alpha=0.1)
    return

def plotObservedMix(dataset,means,sigmas,contours=True):
    plt.figure(figsize=(2,2))
    #colorList = ['darkcyan','hotpink','plum']
    minIdx = int(np.min(dataset[:,2]))
    maxIdx = int(np.max(dataset[:,2]))
    mixtureElements = [pullCluster(dataset,idx) for idx in range(minIdx,maxIdx+1)]
    for idx,element in enumerate(mixtureElements):
        colorIdx = idx % len(colorList)
        if not contours:
            dataScatter(element,color=colorList[colorIdx])
        if contours:
            dataScatter(element)
            plotContour(means[idx],sigmas[idx],plt.gca(),color=colorList[colorIdx])
    return

def plotUnobservedMix(dataset):
    plt.figure(figsize=(2,2))
    minIdx = int(np.min(dataset[:,2]))
    maxIdx = int(np.max(dataset[:,2]))
    mixtureElements = [pullCluster(dataset,idx) for idx in range(minIdx,maxIdx+1)]
    for idx,element in enumerate(mixtureElements):
        dataScatter(element)
    return

def plotEMResults(dataset,K,parameters):
    plt.figure()
    dataScatter(dataset)
    for idx in range(K):
        mu = parameters['mu'][idx]; sigma = parameters['sigma'][idx]
        coloridx = idx % (len(colorList))
        plotContour(mu,sigma,plt.gca(),color=colorList[coloridx],numContours=5)
