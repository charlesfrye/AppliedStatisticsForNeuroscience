import numpy as np

def pull_cluster(dataset,idx):
    indices = np.equal(dataset[:,2],idx)
    cluster_data = dataset[indices,:-1]
    return cluster_data

def run_EM(dataset,old_parameters,update_sigma=True,update_w=True):
    #inference step - Bayes' Rule
    posteriors = infer_hidden_states(dataset,old_parameters)
    #maximization step - weighted version of observed method
    new_parameters = update_parameters(dataset,posteriors,old_parameters,
                                     update_sigma=update_sigma,
                                     update_w=update_w)
    return posteriors,new_parameters

def infer_hidden_states(dataset,parameters):
    #use the parameters to compute the posterior of alpha for each datapoint
    posteriors = []
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
        posteriors.append(compute_posterior(datapoint,parameters))

    return np.asarray(posteriors)

def update_parameters(dataset,posteriors,old_parameters,update_sigma=True,update_w=True):
    #update (some of) the parameters using the "weighted average" rule

    new_parameters = old_parameters #just copying the structure
    num_elements = len(new_parameters['w']) #how many mixture Elements?
    d = len(new_parameters['mu'][0]) #how many dimensions?
    num_datapoints = len(dataset) #how many datapoints

    mu = [np.zeros(d)]*num_elements #initialize a list of num_elements vectors, each with dimension d
    sigma = [np.zeros((d,d))]*num_elements #same as above, but for a list of covariances
    w = [0]*num_elements #and a list of scalars

    for alpha in range(num_elements):

        total_posterior_alpha = sum([posterior[alpha] for posterior in posteriors])

        mu[alpha] = posterior_weighted_average(dataset,posteriors[:,alpha])

        empirical_covariances = calculate_covariances(dataset,mu[alpha])

        if update_sigma == True:
            sigma[alpha] = posterior_weighted_average(empirical_covariances,posteriors[:,alpha])
            new_parameters['sigma'] = sigma

        if update_w == True:
            w[alpha] = total_posterior_alpha/num_datapoints
            new_parameters['w'] = w

    new_parameters['mu'] = mu

    return new_parameters

def calculate_covariances(dataset,mu):
    num_elements = len(dataset)
    diffs = [np.atleast_2d(x-mu) for x in dataset]
    return [ np.dot(diff.T, diff) * 2**-1
            for diff in diffs]

def compute_posterior(x,parameters):
    mu = parameters['mu'] #list of means
    sigma = parameters['sigma'] #list of cov matrices
    w = parameters['w'] # list of p(alpha)s

    num_elements = len(w)

    unnormalized_posteriors = [] #list containing top half of Bayes rule fraction
    for alpha in range(num_elements):
        prior = w[alpha] #p(alpha)
        likelihood = multivariate_gauss_pdf(x,mu[alpha],sigma[alpha]) #p(x|alpha)
        unnormalized_posterior = prior*likelihood #top half of fraction from Bayes rule
        unnormalized_posteriors.append(unnormalized_posterior)

    data_marginal = sum(unnormalized_posteriors) #bottom half of fraction from Bayes rule

    #we want a posterior distribution - p(alpha|x) - so we end up with a list of length num_elements
    posterior = [unnormalized_posterior/data_marginal for unnormalized_posterior in unnormalized_posteriors]

    return posterior

def multivariate_gauss_pdf(x,mu,sigma):
    lamda = np.linalg.inv(sigma) #alternative name for sigma inverse
    d = len(mu)
    Z = np.sqrt(2*np.pi)**(d/2) * np.sqrt(np.linalg.det(sigma))
    diff = x-mu
    p = Z**-1 * np.exp(-0.5 * np.dot(np.dot(diff.T,lamda), diff))
    return p

def posterior_weighted_average(values,posterior):
    count = len(posterior)
    return sum([values[idx]*posterior[idx] for idx in range(count)])/sum(posterior)


def make_params(K=2,mu_spread=2,sigma_width=1):
    #mu_spread -- how big of a square should I draw the means from?
    #sigma_width -- how big should the variances be?

    mu = [np.random.uniform(low=-mu_spread,high=mu_spread,size=2) for _ in range(K)]
    sigma = []
    for _ in range(K+1):
        # initially commented: code to generate random covariance matrix
        #randMat = np.random.uniform(size=(2,2))
        #randPSD = randMat @ randMat.T

        sigma.append(sigma_width*np.eye(2)) #code to just pick a scaled identity matrix
    w = [1/K]*K
    return {'mu': mu, 'sigma':sigma,'w':w}

def setup_K_means(K):
    parameters = make_params(K, #how many components?
                             #means are drawn uniformly from a square centered at 0
                             mu_spread=2, #how big should that square be?
                             #covariance matrices are scaled identity matrix
                             sigma_width=0.01 #what should that scaling factor be?
                                 # make it small to approximate K-Means
                            )

    update_sigma = False
    update_w = False
    return parameters,update_sigma,update_w
