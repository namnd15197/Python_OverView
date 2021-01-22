#load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create adaboost tree classifier object
adaboost = AdaBoostClassifier(random_state = 0)

#train model
model = adaboost.fit(features, target)











