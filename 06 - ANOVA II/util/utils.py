import numpy as np
import pandas as pd

import string

import statsmodels.api as sm
import statsmodels.formula.api as smf

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
