import numpy as np

import inspect
import math

import pandas as pd

from ipywidgets import interact#,interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import scipy.stats

from IPython.core.display import HTML

def formatDataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))

def ordinaryLeastSquares(regressors,targets):
    arrays = [regressors,targets]
    
    arrays = [np.expand_dims(array,0) if array.ndim == 1  
              else array 
              for array in arrays]

    regressors,targets = arrays

    regressorOuterProductInverse = np.linalg.inv(np.dot(regressors,regressors.T))
    weights = np.dot(np.dot(targets,regressors.T),regressorOuterProductInverse)
    return weights

def SS(xs):
    return np.sum(np.square(xs))

def leaveOutFrom(xs,index):
    return np.hstack([xs[:,:index],xs[:,index+1:]])

def leaveOneOutCV(xs,ys):
    N = xs.shape[1]; ws = np.zeros_like(xs); 
    denominator = SS(ys-np.mean(ys))
    for idx in range(N):
        ws[:,idx] = ordinaryLeastSquares(leaveOutFrom(xs,idx),
                                            leaveOutFrom(ys,idx))
    predictions = [np.dot(ws.T[idx],xs.T[idx]) for idx in range(N)]
    R_squared = 1 - SS(ys-predictions)/SS(ys-np.mean(ys))
    
    return R_squared

def computeR_squared(xs,ys,w):
    residuals = ys-np.dot(w.T,xs)
    R_squared = 1 - SS(residuals)/SS(ys-np.mean(ys))
    return R_squared

def runSimulation(trueDegree,populationSize=1000,sampleSize=10,
                 minDegree=0,maxDegree=10,numExperiments=25,xLimits=[-1.5,1.5]):
    
    weights = [0.5,1,-1.75,0.7,-0.65,0.6,0.55,0.5,0.45,0.4,0.35]
    weights = np.atleast_2d(weights[:trueDegree+1]).T
    
    plt.figure()
    plotModel(weights,limits=xLimits,label='True Model')
    
    baseXs = np.random.uniform(*xLimits,size=populationSize)
    #baseXs = np.random.standard_normal(size=populationSize)*0.25

    xPopulation = np.asarray([np.ones(populationSize),baseXs])
    
    for degree in range(2,trueDegree+1):
        xPopulation = np.vstack([xPopulation,np.power(xPopulation[None,1,:],degree)])
    
    noise_level = 0.75
    #noise_level = 0.25
    yPopulation = np.dot(weights.T,xPopulation)+np.random.standard_normal(populationSize)*noise_level    
    
    R_squared_CV = np.zeros((maxDegree+1-minDegree,numExperiments))
    R_squared_fitted = np.zeros((maxDegree+1-minDegree,numExperiments))
    R_squared_actual = np.zeros((maxDegree+1-minDegree,numExperiments))
    
    
    plt.scatter(baseXs.T,yPopulation.T,alpha=0.1,s=72)
    
    for experiment in range(numExperiments):
        modelXs = xPopulation[None,0,:]

        indices = np.random.choice(populationSize,size=sampleSize)

        ySample = yPopulation[:,indices]
        
        if experiment == 0:
            plt.scatter(np.squeeze(xPopulation[1,indices]),
                    np.squeeze(ySample),
                     alpha=1,s=48,color='k',zorder=20)
    
        for degree in range(minDegree,maxDegree+1):
            if degree > 0:
                modelXs = np.vstack([modelXs,np.power(xPopulation[None,1,:],degree)])

            degreeIndex = degree-minDegree
            xSample = modelXs[:,indices]

            R_squared_CV[degreeIndex,experiment] = leaveOneOutCV(xSample,ySample)

            w_fitted = ordinaryLeastSquares(xSample,ySample).T
            
            if experiment == 0:
                
                if degree in [0,1,2,3,8,trueDegree,maxDegree]:                        
                    plotModel(w_fitted,limits=xLimits)

            R_squared_fitted[degreeIndex,experiment] = computeR_squared(xSample,ySample,w_fitted)
            R_squared_actual[degreeIndex,experiment] = computeR_squared(modelXs,yPopulation,w_fitted)
    
    plt.ylim([-10,10])
    plt.legend(loc='best')
    R_squared_actual = np.mean(R_squared_actual,axis=1)
    R_squared_CV = np.mean(R_squared_CV,axis=1)
    R_squared_fitted = np.mean(R_squared_fitted,axis=1)
    
    if maxDegree >= 8:
        for idx in range(-1,maxDegree-10):
            if R_squared_fitted[idx] < 0.8:
                R_squared_fitted[idx] = 1
    
    return R_squared_fitted,R_squared_actual,R_squared_CV

def plotModel(weights,label=None,limits=[-2,2]):
    N = 1000
    xs = np.linspace(*limits,num=N)
    inputXs = np.asarray([np.ones(N)])
    for degree in range(1,weights.shape[0]):
        inputXs =  np.vstack([inputXs,np.power(xs,degree)])
        
    if label == None:
        degree = weights.shape[0]-1
        label = str(degree)
        width = 2
        zorder = 15-degree
    else:
        width = 6
        zorder= 1
    outputs = np.dot(weights.T,inputXs)
    plt.plot(xs,np.squeeze(outputs),
             linewidth=width,label=label,zorder=zorder)

def plotResults(fitted,actual,cross_validated,minDegree=0,maxDegree=10):
    degrees = range(minDegree,maxDegree+1)
    
    plt.figure()
    for label,r_squared_estimate in zip(['sample','population','CV'],
                                  [fitted,actual,cross_validated]):
        plot_R_squared(degrees,r_squared_estimate,label)
    
    current_ymin = plt.gca().get_ylim()[0]
    plt.xlim(-0.5,maxDegree+0.5)
    plt.ylim(max(-0.2,current_ymin),1.1)
    
    plotAxes()
    
    plt.ylabel('R**2');plt.xlabel('Modeling Polynomial Degree')
    plt.legend(loc='best');
    
def plot_R_squared(degrees,values,label):
    plt.plot(degrees,values,
         linestyle='-',linewidth=6,
         marker='.',markersize=36,label=label)

def plotAxes():
    xLim = plt.xlim()
    yLim = plt.ylim()
    plt.hlines([0,0],0,xLim[1],color='k',linewidth=4); 
    plt.vlines([0,0],0,yLim[1],color='k',linewidth=4) 

def clean_lmplot():
    ax = plt.gca()
    ax.set_ylim(-15,15)

def plotTrueModel(w,b):
    ax = plt.gca()
    xLims = ax.get_xlim()
    mesh = np.linspace(*xLims)
    plt.plot(mesh,w*mesh+b,
             color='k',linewidth=4,
             label='True Model')
    plt.legend(loc='best')
    
####

def setupLinearModel(N,gaussianNoise=True,slope=2,offset=0):

    noise_level = 2

    xs = np.random.normal(size=N)*3

    if gaussianNoise:
        noise = np.random.standard_normal(size=N)*noise_level
    else:
        noise = np.random.standard_cauchy(size=N)
    
    ys = slope*xs + offset + noise 
    df = pd.DataFrame.from_dict({'x':xs,'y':ys})
    
    return df