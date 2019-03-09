import numpy as np
import pandas as pd

import string

import statsmodels.api as sm
import statsmodels.formula.api as smf

def generate_data(num_subjects,num_factors,standard_dev):

    outcome = np.random.normal(scale=standard_dev,size=(1,num_subjects))
    predictors = np.random.binomial(1,0.5,size=(num_factors,num_subjects))
    data = np.vstack([outcome,predictors])

    df = {}

    labels = ['outcome']+list(string.ascii_lowercase)

    for factor_data,factor_label in zip(data,labels):
        df[factor_label] = factor_data

    df = pd.DataFrame.from_dict(df)

    return df

def run_ANOVA(df):
    predictors = '*'.join([column for column in df.columns if column is not "outcome"])
    ols_lm = smf.ols('outcome ~ ' + predictors, data=df)
    fit = ols_lm.fit()
    results = sm.stats.anova_lm(fit,typ=2)

    return results
