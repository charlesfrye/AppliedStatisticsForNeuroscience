import numpy as np
import pandas as pd

import string

import statsmodels.api as sm
import statsmodels.formula.api as smf

from IPython.core.display import HTML


def formatDataframes():
    css = open('./css/style-table.css').read()
    return HTML('<style>{}</style>'.format(css))



numSubjects = 10000
numFactors = 7
standardDev = 1


def generateData(numSubjects,numFactors,standardDev):
    
    outcome = np.random.normal(scale=standardDev,size=(1,numSubjects))
    predictors = np.random.binomial(1,0.5,size=(numFactors,numSubjects))
    data = np.vstack([outcome,predictors])
    
    df = {}
    
    labels = ['outcome']+list(string.ascii_lowercase)
    
    for factorData,factorLabel in zip(data,labels):
        df[factorLabel] = factorData
        
    df = pd.DataFrame.from_dict(df)
    
    return df

def runANOVA(df):
    predictors = '*'.join(df.columns[:-1])
    ols_lm = smf.ols('outcome ~ '+predictors,data=df)
    fit = ols_lm.fit()
    results = sm.stats.anova_lm(fit,typ=2)

    return results

nullData = generateData(numSubjects,numFactors,standardDev)

nullData.sample(10)