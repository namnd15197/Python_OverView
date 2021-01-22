#load libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#make class highly imbalanced by removing first 40 observations

features = features[40:,:]
target = target[40:,:]

#create target vector indicating if class 0 otherwise 1
target = np.where((target==0), 0, 1)

#standardize features
scaler = StandardScaler()
features_scaler = scaler.fit_transform(features)

#create decision tree classifier object
logistic_regression = LogisticRegression(random_state = 0, class_weight="balanced")

#train model
model = logistic_regression.fit(features_scaler, target)
