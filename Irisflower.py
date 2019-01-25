# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 09:47:14 2018
@author: Aravinth
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from  random import uniform
import mlflow
import mlflow.sklearn

iris = load_iris()


#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])

test_idx = [0,50,100]

#training data
train_target  = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx] 
test_data = iris.data[test_idx]


clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#Predicting test data
print(test_target)
print(clf.predict(test_data))

#print("test_data",test_data[0])
#print("test_target",test_target[0])
#print(iris.feature_names,iris.target_names)

#print(clf.predict([[4.96, 3.01, 1.42, 0.23]]))



for i in range(20):
    sepal_length = uniform(4,8)
    mlflow.log_param("sepal_length", sepal_length)
    sepal_width = uniform(2,4.6)
    mlflow.log_param("sepal_width", sepal_width)
    petal_length = uniform(1,7.1)
    mlflow.log_param("petal_length", petal_length)
    petal_width = uniform(0,2.6)
    mlflow.log_param("petal_width", petal_width)
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    mlflow.log_metric("flower type", prediction[0])
print(prediction)
