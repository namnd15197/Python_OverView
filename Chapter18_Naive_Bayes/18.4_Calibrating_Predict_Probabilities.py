#load libraries
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
#create gaussian naive bayes object 
classifier = GaussianNB()

#create calibrated cross-validation with sigmoid calibration
classifier_sigmoid = CalibratedClassifierCV(classifier, cv=2, method='sigmoid')

#calibrate probabilities
classifier_sigmoid.fit(features, target)

#create new observation
new_observation = [[2.6, 2.6, 2.6, 0.4]]

#view calibrated probabilities
print(classifier_sigmoid.predict_proba(new_observation))

#train a gassian naive bayes then predict class probabilities
print(classifier.fit(features, target).predict_proba(new_observation))

#view calibrated probabilities
print(classifier_sigmoid.predict_proba(new_observation))

