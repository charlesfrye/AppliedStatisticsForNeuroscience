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
      
def randomWeights(d=1):
    return np.random.standard_normal(size=(d,1))

def plotModel(x,y):
    plt.figure()
    plt.plot(np.squeeze(x),np.squeeze(y),linewidth=4)

def setupX(N,xMode='linspace',xRange=[-2,2]):
    if xMode == 'uniform':
        x = uniformInputs(xRange[0],xRange[1],N)
    elif xMode == 'gauss':
        xWidth = xRange[1] - xRange[0]
        mu = (xRange[1] + xRange[0])/2
        sd = xWidth/3
        x = gaussInputs(mu,sd,N)
    elif xMode == 'linspace':
        x = linspaceInputs(xRange[0],xRange[1],N)
    else:
        print("mode unrecognized, defaulting to linspace")
        x = linspaceInputs(-1,1,N)
        
    return x

def randomLinearModel(noise_level,xMode='linspace',N=1000):
    if xMode == 'uniform':
        x = uniformInputs(-1,1,N)
    elif xMode == 'gauss':
        x = gaussInputs(0,1,N)
    elif xMode == 'linspace':
        x = linspaceInputs(-1,1,N)
    else:
        print("mode unrecognized, defaulting to linspace")
        x = linspaceInputs(-1,1,N)
        
    allOnes = np.ones(N)
    regressors = np.vstack([x,allOnes])
    
    linearWeights = randomWeights(2)
    
    epsilon = noise_level*np.random.standard_normal(size=(1,N))
    
    linearY = np.dot(linearWeights.T,regressors) + epsilon
    
    linearModelDataFrame = pd.DataFrame.from_dict({'x':np.squeeze(x),'y':np.squeeze(linearY)})
    
    return linearModelDataFrame

def randomLinearizedModel(noise_level,maxDegree,xMode='linspace',xRange=[-1,1],N=1000):
    
    x = setupX(N,xMode=xMode,xRange=xRange)
        
    allOnes = np.ones(N)
    
    polyRegressors = [np.power(x,n) for n in range(2,maxDegree+1)]
    
    regressors = np.vstack([x,allOnes]+polyRegressors)
    
    weights = randomWeights(maxDegree+1)
    
    epsilon = noise_level*np.random.standard_normal(size=(1,N))
    
    linearY = np.dot(weights.T,regressors) + epsilon
    
    linearizedModelDataFrame = pd.DataFrame.from_dict({'x':np.squeeze(x),'y':np.squeeze(linearY)})
    
    return linearizedModelDataFrame

def randomNonlinearModel(noise_level,function,
                       xMode='linspace',N=1000,
                       xRange = [-2,2],
                      thetaRange=[-1,1]):
   
    x = setupX(N,xMode=xMode,xRange=xRange)
    
    theta = setupTheta(thetaRange)
    
    epsilon = noise_level*np.random.standard_normal(size=(1,N))
    
    nonlinearY = function(theta,x) + epsilon
    
    nonlinearModelDataFrame = pd.DataFrame.from_dict({'x':np.squeeze(x),
                                                      'y':np.squeeze(nonlinearY)})
    
    return nonlinearModelDataFrame

def uniformInputs(mn,mx,N):
    return np.random.uniform(mn,mx,size=(1,N))

def gaussInputs(mn,sd,N):
    return mn+sd*np.random.standard_normal(size=(1,N))

def linspaceInputs(mn,mx,N):
    return np.linspace(mn,mx,N)

def makeNonlinearTransform(transform,thetaFirst=True):
    if thetaFirst:
        return lambda theta,x: transform(theta,x)
    else:
        return lambda theta,x: transform(x,theta)

def makePowerTransform():
    return makeNonlinearTransform(np.power,thetaFirst=False)

def makeLNTransform(f):
    """linear-nonlinear transforms"""
    return lambda theta,x: f(theta*x)

def makeNonlinearParameters(default,rangeTuple):
    return Parameters([default],[rangeTuple],['theta'])

def makeRectLinTransform():
    return lambda theta,x: np.where(x>theta,x-theta,0)

def setupTheta(thetaRange):
    thetaWidth = thetaRange[1] - thetaRange[0]
    theta = np.random.rand()*thetaWidth+thetaRange[0]
    return theta

def ordinaryLeastSquares(regressors,targets):
    arrays = [regressors,targets]
    
    arrays = [np.expand_dims(array,0) if array.ndim == 1  
              else array 
              for array in arrays]

    regressors,targets = arrays

    regressorOuterProductInverse = np.linalg.inv(np.dot(regressors,regressors.T))
    weights = np.dot(np.dot(targets,regressors.T),regressorOuterProductInverse)
    return weights

def axisEqual3D(ax,center=0):
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

## Computing and Plotting residuals and regressor-residual correlations

def runSimulation(numExperiments,numDatapoints,useGroundTruth,gaussianNoise,noise_level):

    trueW = 2

    rs = []

    for _ in range(numExperiments):

        xs = np.asarray(np.random.standard_normal(numDatapoints))

        if gaussianNoise:
            noise = np.random.standard_normal(numDatapoints)
        else:
            noise = np.multiply(np.random.choice([-1,1],size=numDatapoints),
                                np.random.standard_exponential(numDatapoints))

        noise = noise*noise_level

        ys = trueW*xs+noise

        if useGroundTruth:
            w = trueW
        else:
            w = ordinaryLeastSquares(xs,ys)

        residuals = ys-w*xs

        #all_residuals = np.hstack([all_residuals,np.squeeze(residuals)])
        rs.append(np.corrcoef(residuals,xs)[0,1])
        
    return xs,ys, rs, residuals

def plotSimulationResults(exampleXs,exampleYs,rs,exampleResiduals):

    plt.figure()    
    ax = sns.distplot(rs,hist=True,kde=False,rug=False,
                 hist_kws={'histtype':'stepfilled','linewidth':4,'normed':True});
    ax.set_xlim([-1,1])
    plt.title('Distribution of Correlations Between Residuals and Regressors')

    plt.figure()
    plt.scatter(exampleXs,exampleYs)
    plt.title('Example Dataset')


    plt.figure()
    sns.distplot(exampleResiduals,hist=True,kde=False,rug=False,
                 hist_kws={'histtype':'stepfilled','linewidth':4,'normed':True})
    plt.title('Example Distribution of Residuals');
    
    return

def doLeastSquares(numDatapoints):
    useGroundTruth = False
    
    xs = np.asarray(np.random.standard_normal(numDatapoints))

    if gaussianNoise:
        noise = np.random.standard_normal(numDatapoints)
    else:
        noise = np.multiply(np.random.choice([-1,1],size=numDatapoints),
                                np.random.standard_exponential(numDatapoints))

    noise = noise*noise_level

    ys = trueW*xs+noise

    if useGroundTruth:
        w = trueW
    else:
        w = ordinaryLeastSquares(xs,ys)

    residuals = ys-w*xs

