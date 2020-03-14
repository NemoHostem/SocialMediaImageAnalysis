# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:47:36 2020

@author: Matias
"""

#%% 1. Load data

import numpy as np
#from sklearn import preprocessing
#from sklearn.model_selection import GroupShuffleSplit
#from sklearn import discriminant_analysis
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import neighbors, svm, linear_model
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
#import pandas as pd

path = ''

X_train_orig = X
y_train_orig = []

for d in data:
    y_train_orig.append(d[2][-1])


#%% 2. Split to training and testing.

#groups = pd.groupby(y_train_orig)
res_MSE = np.empty([10,10])
res_R2 = np.empty([10,10])

for r in range(10):

    X_train, X_test, y_train, y_test = X_train_orig[:10000], X_train_orig[10000:12500], y_train_orig[:10000], y_train_orig[10000:12500]

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)   

    regressors= [neighbors.KNeighborsRegressor(n_neighbors=1),
                  neighbors.KNeighborsRegressor(n_neighbors=5),
                  neighbors.KNeighborsRegressor(n_neighbors=10),
                  #svm.SVR(kernel="linear"),
                  #svm.SVR(kernel="rbf"),
                  svm.LinearSVR(max_iter = 5000),
                  linear_model.LinearRegression(),
                  RandomForestRegressor(n_estimators=1000),
                  AdaBoostRegressor(n_estimators=1000),
                  ExtraTreesRegressor(n_estimators=1000),
                  GradientBoostingRegressor(n_estimators=1000)]
    
    #regressor_names = ["1-NN","5-NN","Linear SVR","RBF SVR", "Linear SVR", "Linear Regression","RandomForest","AdaBoost","Extra Trees","GB Regressor"]
    regressor_names = ["1-NN","5-NN","10-NN","Linear SVR","Linear Regression","RandomForest","AdaBoost","Extra Trees","GB Regressor"]
    
    
    for i, regressor in enumerate(regressors):
        print(regressor_names[i])
        #try:
        regressor.fit(X_train,y_train)
        res_MSE[i][r] = (mean_squared_error(y_test, regressor.predict(X_test)))
        res_R2[i][r] = (r2_score(y_test, regressor.predict(X_test)))
        print(regressor_names[i]+": MSE: "+str(res_MSE[i][r])+", R2: "+str(res_R2[i][r])+".")
        #except ValueError:
        #print(classifiers_names[i])
    
#%% Print mean, min and max of each classifier
        
for n, name in enumerate(regressor_names):
    print("Mean of " + name + ": MSE: " + str(np.mean(res_MSE[n][:])) + ", R2: "  + str(np.mean(res_R2[n][:])) + ".")
    print("Min of " + name + ": MSE: " + str(np.min(res_MSE[n][:])) + ", R2: "  + str(np.min(res_R2[n][:])) + ".")
    print("Max of " + name + ": MSE: " + str(np.max(res_MSE[n][:])) + ", R2: "  + str(np.max(res_R2[n][:])) + ".")


#%%
    
import matplotlib.pyplot as plt

plt.bar(regressor_names, [np.mean(res_R2[m][:]) for m in range(9)])
plt.xlabel('Method')
plt.ylabel('R2 of 2500 samples')
plt.title('Machine learning regressors')
plt.grid(True)
plt.show()