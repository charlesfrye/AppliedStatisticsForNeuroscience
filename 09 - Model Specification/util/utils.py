import numpy as np

import inspect
import math

import pandas as pd

from ipywidgets import interact#,interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt

def randomWeights(d=1):
    return np.random.standard_normal(size=(d,1))

def plot_model(x, y):
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

def setupTrig(trigFunction,thetaRange=[-5,5]):
    inputValues = np.linspace(-10,10,200)
    
    parameters = makeNonlinearParameters(1,thetaRange)
    
    transform = makeLNTransform(trigFunction)
    
    return inputValues,parameters,transform

def setupPower(maxDegree):
    inputValues = np.linspace(0,10,200)
    
    parameters = makeNonlinearParameters(1,[0,maxDegree])
    
    transform = makePowerTransform()
    
    return inputValues,parameters,transform

def setupLN(f,inputRange,thetaRange=[-1,1]):
    inputValues = np.linspace(*inputRange,num=200)
    
    parameters = makeNonlinearParameters(0,thetaRange)
    
    transform = makeLNTransform(f)
    
    return inputValues,parameters,transform

def setupRectLin(thetaRange=[-10,10]):
    
    transform = makeRectLinTransform()

    inputValues = np.linspace(-10,10,200)

    parameters = makeNonlinearParameters(0,thetaRange)
    
    return inputValues,parameters,transform

def setupTheta(thetaRange):
    thetaWidth = thetaRange[1] - thetaRange[0]
    theta = np.random.rand()*thetaWidth+thetaRange[0]
    return theta

