# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:47:36 2020

@author: Matias
"""

#%% 1. Load data

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from sklearn import discriminant_analysis
from sklearn.metrics import accuracy_score
from sklearn import neighbors, svm, linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import pandas as pd

path = ''

"""
X_train_orig = np.load(path+"X_train_kaggle.npy")
X_kaggle_test = np.load(path+"X_test_kaggle.npy")
y_train_data = np.genfromtxt(path+"groups.csv", delimiter=",", dtype=[("id",np.uint),("group_id",np.uint),("surface","S22")])
y_train_orig = y_train_data["surface"]
"""
X_train_orig = X
y_train_orig = []

for d in data:
    y_train_orig.append(d[2][-1])

#%% 2. Create an index of class names.

le = preprocessing.LabelEncoder()
le.fit(y_train_orig)
y_train = le.transform(y_train_orig)

#%% 3. Split to training and testing.

#groups = pd.groupby(y_train_orig)
res = np.empty([10,100])

for r in range(100):

    #train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=r).split(X_train_orig, groups=groups))
    X_train, X_test, y_train, y_test = X_train_orig[:int(len(X_train_orig)/2)], X_train_orig[int(len(X_train_orig)/2):], y_train_orig[:int(len(y_train_orig)/2)], y_train_orig[int(len(y_train_orig)/2):]


    
    X_train = np.array(X_train).astype(int)
    X_test = np.array(X_test).astype(int)
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    """
    X_train2 = np.mean(X_train1,axis=2)
    X_test2 = np.mean(X_test1,axis=2)


    X_train = np.std(X_train,axis=2)
    X_train = np.concatenate((X_train2, X_train), axis=1)
    
    X_test = np.std(X_test,axis=2)
    X_test = np.concatenate((X_test2, X_test), axis=1)
    """
    
    

    classifiers= [neighbors.KNeighborsClassifier(n_neighbors=1),
                  neighbors.KNeighborsClassifier(n_neighbors=5),
                  discriminant_analysis.LinearDiscriminantAnalysis(),
                  svm.SVC(kernel="linear"),
                  svm.SVC(kernel="rbf",gamma="auto"),
                  linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial",max_iter=2000),
                  RandomForestClassifier(n_estimators=1000),
                  AdaBoostClassifier(),
                  ExtraTreesClassifier(n_estimators=1000),
                  GradientBoostingClassifier()]
    
    classifiers_names = ["1-NN","5-NN","LDA","Linear SVC","RBF SVC","Logistic Regression","RandomForest","AdaBoost","Extra Trees","GB-Trees"]
    
    for i, classifier in enumerate(classifiers):
        #try:
        classifier.fit(X_train,y_train)
        res[i][r] = (accuracy_score(y_test, classifier.predict(X_test)))
        print(classifiers_names[i]+": "+str(100*accuracy_score(y_test, classifier.predict(X_test)))+" %")
        #except ValueError:
        print(classifiers_names[i])
    
#%% Print mean, min and max of each classifier
        
for n, name in enumerate(classifiers_names):
    print("Mean of " + name + ": " + str(100*np.mean(res[n][:])) + " %")
    print("Min of " + name + ": " + str(100*np.min(res[n][:])) + " %")
    print("Max of " + name + ": " + str(100*np.max(res[n][:])) + " %")

'''
import xgboost as xgb

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
'''

#%% Create submission file

''' RandomForest chosen as the classifier

classifier = classifiers[6]

y_pred = classifier.predict(X_kaggle_test)
labels = list(le.inverse_transform(y_pred))
with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        print (str(label)[2:-1])
        fp.write("%d,%s\n" % (i, str(label)[2:-1]))

'''
#%%