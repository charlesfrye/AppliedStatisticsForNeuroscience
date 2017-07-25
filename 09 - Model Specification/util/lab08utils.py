import numpy as np

import inspect
import math

import pandas as pd

from ipywidgets import interact#,interactive, fixed, interact_manual

import ipywidgets as widgets

import matplotlib.pyplot as plt

from IPython.core.display import HTML

def formatDataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))

class Parameters(object):
    
    def __init__(self,defaults,ranges,names=None):
        assert len(defaults) == len(ranges), "must have default and range for each parameter"
        
        self.values = np.atleast_2d(defaults)
        
        self.num = len(defaults)
        
        self._zip = zip(defaults,ranges)
        
        if names == None:
            self.names = ['parameter_'+str(idx) for idx in range(self.num)]
        else:
            self.names = names
        
        # if len(self.values.shape) > 1:
        #     values_for_dict = np.squeeze(self.values)
        # else:
        #     values_for_dict = self.values
        
        self.dict = dict(zip(self.names,self.values))
        #self.dict = dict(zip(self.names,np.squeeze(self.values)))
        
        self.defaults = defaults
        self.ranges = ranges
        
        self.makeWidgets()
        
    def makeWidgets(self):
        self._widgets = [self.makeWidget(parameter,idx) 
                        for idx,parameter
                        in enumerate(self._zip)]
        
        self.widgets = {self.names[idx]:_widget
                        for idx,_widget
                       in enumerate(self._widgets)}
    
    def makeWidget(self,parameter,idx):
        default = parameter[0]
        range = parameter[1]
        name = self.names[idx]
        return widgets.FloatSlider(value=default,
                                    min= range[0],
                                   max = range[1],
                                   step = 0.01,
                                   description=name
                                    )
    
    def update(self):
        sortedKeys = sorted(self.dict.keys())
        self.values = np.atleast_2d([self.dict[key] for key in sortedKeys])
        
        
class Model(object):
    
    def __init__(self,inputValues,modelInputs,parameters,funk):
        self.modelInputs = np.atleast_2d(modelInputs)
        self.inputValues = inputValues
        self.parameters = parameters
        self.funk = funk
        self.plotted = False
        
    def plot(self):
        if not self.plotted:
            self.initializePlot()
        else:
            self.artist.set_data(self.inputValues,self.outputs)
        return
    
    @property
    def outputs(self):
        return np.squeeze(self.funk(self.modelInputs))
    
    def initializePlot(self):
        self.fig = plt.figure()
        self.ax = plt.subplot(111)
        self.artist, = plt.plot(self.inputValues,
                                self.outputs,
                               linewidth=4)
        self.plotted = True
        self.ax.set_ylim([-10,10])
        self.ax.set_xlim([-10,10])
    
    def makeInteractive(self):
        @interact(**self.parameters.widgets)
        def make(**kwargs):
            frame = inspect.currentframe()
            args, _, _, values = inspect.getargvalues(frame)
            kwargs = values['kwargs']
            for parameter in kwargs.keys():
                self.parameters.dict[parameter] = kwargs[parameter]
            self.parameters.update()
            self.plot()
            return
        
        return
    
class LinearModel(Model):
    
    def __init__(self,inputValues,parameters,modelInputs=None):
        
        if modelInputs == None:
            modelInputs = self.transformInputs(inputValues)
        else:
            modelInputs = modelInputs
        
        def funk(inputs):
            return np.dot(self.parameters.values,inputs)
        
        Model.__init__(self,inputValues,modelInputs,parameters,funk)
        
    def transformInputs(self,inputValues):
        modelInputs = [[1]*inputValues.shape[0],inputValues]
        return modelInputs
    
class LinearizedModel(LinearModel):
    
    def __init__(self,transforms,inputValues,parameters):
        
        self.transforms = [lambda x: np.power(x,0), 
                           lambda x: x] + transforms
        
        modelInputs = self.transformInputs(inputValues)
            
        LinearModel.__init__(self,inputValues,parameters,modelInputs=modelInputs)
        
    def transformInputs(self,inputValues):
        transformedInputs = []
        
        for transform in self.transforms:
            transformedInputs.append(transform(inputValues))
            
        return transformedInputs
    
class NonlinearModel(Model):
    
    def __init__(self,inputValues,parameters,transform):
        
        def funk(inputs):
            return transform(self.parameters.values,inputs)
        
        Model.__init__(self,inputValues,inputValues,parameters,funk)

def makeDefaultParameters(number,rnge=1,names=None):
    defaults = [0]*number
    ranges = [[-rnge,rnge]]*number
    return Parameters(defaults,ranges,names)

def makeSineParameters(degree):
    defaults = [(-1)**(n//2)/math.factorial(n) 
                if n%2 != 0 
                else 0 
                for n in range(degree+1)]
    ranges = [[-1,1]]*(degree+1)
    return Parameters(defaults,ranges,names=makePolynomialParameterNames(degree))

def makePolynomialTransforms(maxDegree):
    curried_power_transforms = [lambda n: lambda x: np.power(x,n) for _ in range(2,maxDegree+1)]
    transforms = [curried_power_transform(n) 
                              for curried_power_transform,n
                                  in zip(curried_power_transforms,range(2,maxDegree+1))
                 ]
    return transforms

def makePolynomialParameters(maxDegree,rnge=1):
    return makeDefaultParameters(maxDegree+1,rnge=rnge,
                                 names=makePolynomialParameterNames(maxDegree))

def makePolynomialParameterNames(maxDegree):
    return ['x^'+str(n) for n in range(maxDegree+1)]

def makeLinearParameters():
    return makePolynomialParameters(1)

def makeLinearizedParameters(transforms):
    return makeDefaultParameters(len(transforms)+2)

def makeTrigTransform(f):
    return lambda theta,x: f(theta*x)


def makeNonlinearModelParameters(default,rangeTuple):
    return Parameters(default,rangeTuple,['theta'])

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

