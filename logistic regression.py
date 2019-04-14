# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:44:43 2019

Notebooks : Logistic regression

NOTES :
    
A. LR.score(X,Y) internally calculates Y'=predictor.predict(X) and then compares Y' against Y to give an accuracy measure. This applies not only to logistic regression but to any other model.

B. LR.score(x_train, y_train) is measuring the accuracy of the model against the training data. (How well the model explains the data it was trained with). <-- But note that this has nothing to do with test data.

C. LR.score(x_test, y_test) is equivalent to your print(classificationreport(y_test, y_pred)). But you do not need to calculate Ypred; that is done internally by the library

"""
#%% STEP 1 : Import libraries and files
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#%% STEP 2 : Load tic-tac-toe data
data = pd.read_csv("tic-tac-toe.data", sep=",")
data.rename(columns={'x': 'top left', 'x.1': 'top middle', 'x.2': 'top right',
                     'x.3': 'middle left', 'o': 'middle middle', 'o.1' : 'middle right', 
                     'x.4' : 'bottom left', 'o.2' : 'bottom middle', 'o.3':'bottom right',
                     'positive' : 'outcome'},inplace=True)

#%% STEP 3 : Data preprocessing
data_new = pd.get_dummies(data.ix[:,0:9])
data_final = pd.concat([data_new, data.ix[:,9]], axis=1)

#%% STEP 4 : Split data into training and testing sets
train, test = train_test_split(data_final, test_size = 0.3)

#%% STEP 5 : Applying logistic regressing model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
x_train = train.ix[:, :-1]
y_train = train.ix[:,-1]
x_test = test.ix[:,:-1]
y_test = test.ix[:,-1]
LR = LR.fit(x_train, y_train)   # Fit the model according to the given training data.
LR.score(x_train, y_train)   # Return the mean accuracy on the given train data and labels.

#%% STEP 6 : Evaluate model on test data
# Finding the accuracy using confusion matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
#prob = LR.predict_proba(x_test)
y_pred = LR.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

# Finding the accuracy using cross validation method
scores = cross_val_score(LogisticRegression(), data_final.ix[:,0:27], data_final.ix[:,27], scoring='accuracy', cv=10)
print(scores)
print(scores.mean())













