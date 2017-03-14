import numpy as np
import scipy.stats

import matplotlib.pyplot as plt

import time

from IPython.core.display import HTML

def formatDataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))

def makeECDF(data):
    
    def eCDF(x):
        N = len(data)
        totalBelow = data <= x
        CD = np.sum(totalBelow)/N
        return CD

    return eCDF

def plotSamples(sampler=np.random.standard_normal,
               trueCDF=scipy.stats.norm.cdf,
               inputRange = np.arange(-5,5,0.1),
               sampleSize = 50,
               numSamples = 10):
    
        
    if numSamples >= 25:
        alpha = 0.25
    else:
        alpha=0.4
    
    fig = plt.figure()
    
    plt.plot(inputRange,trueCDF(inputRange),
            linewidth=8,
             color='k',label='true CDF')
    
    plt.legend(loc='lower right')
    
    plt.xlabel('x',fontsize=24)
    plt.ylabel('P(X ≤ x)',fontsize=24)
    
    fig.canvas.draw()
    time.sleep(0.75)
        
    for idx in range(numSamples):
        
        samples = sampler(size=sampleSize)
        
        eCDF = makeECDF(samples)
        
        plt.plot(inputRange,
                [eCDF(input) for input in inputRange],
                linewidth=2,alpha=alpha,
                color = 'hotpink',label='single sample eCDF')
        
        if idx == 0:
            plt.legend(loc='lower right')
            
        fig.canvas.draw()

def plotConvergingCDF(sampler=np.random.standard_normal,
                      trueCDF=scipy.stats.norm.cdf,
                      inputRange = np.arange(-5,5,0.1),
                      sampleSizes = [1,2,3,5,
                                     10,20,30,50,
                                     100,200,300,500,
                                     1000]):

    fig = plt.figure()
    samples = np.asarray([])
    
    plt.plot(inputRange,trueCDF(inputRange),
            linewidth=8,
             color='k',label='true CDF')
    
    plt.legend(loc=4)
    
    plt.xlabel('x',fontsize=24)
    plt.ylabel('P(X ≤ x)',fontsize=24)
    lastPlottedLine = False

    for idx,sampleSize in enumerate(sampleSizes):
        fig.canvas.draw()
        time.sleep(0.75)
        
        if lastPlottedLine:
            lastPlottedLine.set_visible(False)

        currentSize = len(samples)

        newSamples = sampler(size=sampleSize-currentSize)
        samples = sorted(np.hstack([samples,newSamples]))
        eCDF = makeECDF(samples)

        lastPlottedLine, = plt.plot(inputRange,
                                     [eCDF(input) for input in inputRange],
                                    linewidth=4,#alpha=.8,
                                    color = 'chartreuse',
                                   label = 'sample eCDF')
        plt.title('Sample Size = '+ str(sampleSize))
        
        if idx == 0:
            plt.legend(loc=4)

def plotBootstraps(sampler=np.random.standard_normal,
                    trueCDF=scipy.stats.norm.cdf,
                    inputRange = np.arange(-5,5,0.1),
                    sampleSize = 50,
                    numBootstraps = 10) :
    
    if numBootstraps>25:
        alpha = 0.25
    else:
        alpha=0.4
    
    fig = plt.figure()
    samples = sorted(sampler(size=sampleSize))
    
    eCDF = makeECDF(samples)
    
    plt.plot(inputRange,trueCDF(inputRange),
            linewidth=8,
             color='k',label='true CDF')
    
    plt.legend(loc='lower right')
    
    plt.xlabel('x',fontsize=24)
    plt.ylabel('P(X ≤ x)',fontsize=24)
    
    fig.canvas.draw()
    time.sleep(0.75)
    
    plt.plot(inputRange,[eCDF(input) for input in inputRange],
    linewidth=8,
    color='chartreuse',label='empirical CDF')
    
    plt.legend(loc='lower right')
    
    fig.canvas.draw()
    time.sleep(0.75)
        
    for idx in range(numBootstraps):
        
        bootSamples = np.random.choice(samples,size=len(samples))
        
        bootCDF = makeECDF(bootSamples)
        
        plt.plot(inputRange,
                [bootCDF(input) for input in inputRange],
                linewidth=1,alpha=alpha,
                color = 'hotpink',label='single bootstrap CDF')
        
        if idx == 0:
            plt.legend(loc='lower right')
        fig.canvas.draw()