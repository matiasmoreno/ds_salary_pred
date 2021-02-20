# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:51:32 2021

@author: matias
"""

# Data Modeling

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('salary_data_cleaned.csv')

# Choose relevant columns
df.columns

df_model = df[['salary_avg','company_name_encoded', 'exp_avg', 'description_length',
               'key_skills_length', 'location_abbreviation',
               'job_simplified', 'jd_senior', 'sk_python', 'sk_excel', 'sk_r',
               'sk_ML', 'sk_spark',
               'sk_hadoop', 'sk_sql']]
# Get Dummy data, for categorical values
df_dum = pd.get_dummies(df_model)

# Get train and test set
from sklearn.model_selection import train_test_split

X = df_dum.drop('salary_avg', axis=1)
y = df_dum['salary_avg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# lasso regression, sparse variables
lm_l = Lasso()
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error'))

alpha = []
error = []

for i in range (1,200):
    alpha.append(i/1000)
    lml = Lasso(alpha=i/1000)
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring = 'neg_mean_absolute_error')))

plt.plot(alpha,error)

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# tune models GridsearchCV
from sklearn.model_selection import GridSearchCV
'''param_grid = {
    'n_estimators':range(10,100,10),
    'criterion':('mse', 'mae'), 
    'max_features':('auto', 'sqrt', 'log2')
}
gs = GridSearchCV(rf, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)'''

# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lm_l = lm_l.predict(X_test)
tpred_rf = rf.predict(X_test)
#tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lm_l)
mean_absolute_error(y_test, tpred_rf)

# Save model
import pickle
pickl = {'model': rf}
pickle.dump(pickl, open('model_rf_file' + '.p', 'wb'))

# Test import model
model_file = 'model_rf_file.p'
with open(model_file, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
    pred_rf = model.predict(X_test)
    mean_absolute_error(y_test, pred_rf)
# Test one instance
X_test.iloc[1,:].values.reshape(1, -1)
#list(X_test.iloc[1,:])
model.predict(X_test.iloc[1,:].values.reshape(1, -1))

print('Done!')