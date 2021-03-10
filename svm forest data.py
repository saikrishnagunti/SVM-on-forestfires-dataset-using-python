# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:45:46 2021

@author: shivani
"""

import pandas as pd
import numpy as np
forestfires = pd.read_csv("C:\\Users\\shivani\\Desktop\\data science\\module 24 deep learning black box technique - SVM\\forestfires.csv")

forestfires.describe()

##Dropping the month and day columns
forestfires.drop(["month","day"],axis=1,inplace =True)

##Normalising the data as there is scale difference
predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]

def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

fires = norm_func(predictors)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)

model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test) 

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) 

#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test)


