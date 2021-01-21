
#load libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

#load data
iris = datasets.load_iris()

features = iris.data
target = iris.target

#make class highly imbalanced by removing first 40 observations
features = features[40:,:]
target = target[40:]

#create target vector indicating if class 0, otherwise 1
target = np.where((target==0), 0, 1)

#create random forest classifier object
randomforest = RandomForestClassifier(
    random_state = 0, n_jobs=-1, class_weight="balanced")

#train model
model = randomforest.fit(features, target)

observation = [[5, 4, 3, 2]]

print(model.predict(observation))