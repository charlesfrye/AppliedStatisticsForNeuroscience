import numpy as np

def pullCluster(dataset,idx):
    indices = np.equal(dataset[:,2],idx)
    clusterData = dataset[indices,:-1]
    return clusterData

def runEM(dataset,oldParameters,updateSigma=True,updateW=True):
    #inference step - Bayes' Rule
    posteriors = inferHiddenStates(dataset,oldParameters)
    #maximization step - weighted version of observed method
    newParameters = updateParameters(dataset,posteriors,oldParameters,
                                     updateSigma=updateSigma,
                                     updateW=updateW)
    return posteriors,newParameters

def inferHiddenStates(dataset,parameters):
    #use the parameters to compute the posterior of alpha for each datapoint
    posteriors = []
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        posteriors.append(computePosterior(datapoint,parameters))
        
    return np.asarray(posteriors)

def updateParameters(dataset,posteriors,oldParameters,updateSigma=True,updateW=True):
    #update (some of) the parameters using the "weighted average" rule
    
    newParameters = oldParameters #just copying the structure
    numElements = len(newParameters['w']) #how many mixture Elements?
    d = len(newParameters['mu'][0]) #how many dimensions?
    numDatapoints = len(dataset) #how many datapoints
   
    mu = [np.zeros(d)]*numElements #initialize a list of numElements vectors, each with dimension d
    sigma = [np.zeros((d,d))]*numElements #same as above, but for a list of covariances
    w = [0]*numElements #and a list of scalars

    for alpha in range(numElements):
        
        totalPosteriorAlpha = sum([posterior[alpha] for posterior in posteriors])
        
        mu[alpha] = posteriorWeightedAverage(dataset,posteriors[:,alpha])
        
        empiricalCovariances = calculateCovariances(dataset,mu[alpha])
        
        if updateSigma == True:
            sigma[alpha] = posteriorWeightedAverage(empiricalCovariances,posteriors[:,alpha])
            newParameters['sigma'] = sigma

        if updateW == True:
            w[alpha] = totalPosteriorAlpha/numDatapoints
            newParameters['w'] = w
    
    newParameters['mu'] = mu
    
    return newParameters

def calculateCovariances(dataset,mu):
    numElements = len(dataset)
    diffs = [np.atleast_2d(x-mu) for x in dataset]
    return [ np.dot(diff.T, diff) * 2**-1
            for diff in diffs]

def computePosterior(x,parameters):
    mu = parameters['mu'] #list of means
    sigma = parameters['sigma'] #list of cov matrices
    w = parameters['w'] # list of p(alpha)s
    
    numElements = len(w)
    
    unNormalizedPosteriors = [] #list containing top half of Bayes rule fraction
    for alpha in range(numElements):
        prior = w[alpha] #p(alpha)
        likelihood = multivariateGaussPDF(x,mu[alpha],sigma[alpha]) #p(x|alpha)
        unNormalizedPosterior = prior*likelihood #top half of fraction from Bayes rule
        unNormalizedPosteriors.append(unNormalizedPosterior)
        
    dataMarginal = sum(unNormalizedPosteriors) #bottom half of fraction from Bayes rule
    
    #we want a posterior distribution - p(alpha|x) - so we end up with a list of length numElements
    posterior = [unNormalizedPosterior/dataMarginal for unNormalizedPosterior in unNormalizedPosteriors]
    
    return posterior

def multivariateGaussPDF(x,mu,sigma):
    lamda = np.linalg.inv(sigma) #alternative name for sigma inverse
    d = len(mu)
    Z = np.sqrt(2*np.pi)**(d/2) * np.sqrt(np.linalg.det(sigma))
    diff = x-mu
    p = Z**-1 * np.exp(-0.5 * np.dot(np.dot(diff.T,lamda), diff))
    return p

def posteriorWeightedAverage(values,posterior):
    count = len(posterior) 
    return sum([values[idx]*posterior[idx] for idx in range(count)])/sum(posterior)


def makeParams(K=2,muSpread=2,sigmaWidth=1):
    #muSpread -- how big of a square should I draw the means from?
    #sigmaWidth -- how big should the variances be?
    
    mu = [np.random.uniform(low=-muSpread,high=muSpread,size=2) for _ in range(K)]
    sigma = []
    for _ in range(K+1):
        # initially commented: code to generate random covariance matrix
        #randMat = np.random.uniform(size=(2,2))
        #randPSD = randMat @ randMat.T
        
        sigma.append(sigmaWidth*np.eye(2)) #code to just pick a scaled identity matrix
    w = [1/K]*K
    return {'mu': mu, 'sigma':sigma,'w':w}

def setupKMeans(K):
    parameters = makeParams(K, #how many components?
                             
                             #means are drawn uniformly from a square centered at 0
                             muSpread=2, #how big should that square be?
                             
                             #covariance matrices are scaled identity matrix
                             sigmaWidth=0.01 #what should that scaling factor be?
                                 # make it small to approximate K-Means
                            )
    
    updateSigma = False
    updateW = False
    return parameters,updateSigma,updateW
