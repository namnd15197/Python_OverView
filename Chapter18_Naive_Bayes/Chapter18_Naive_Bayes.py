#18.1 Training a classifier for continouos features

#load libraries
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create gaussian naive bayes object
classifier = GaussianNB()
#train model
model = classifier.fit(features, target)

#create new observation
new_observation = [[4, 4, 4, 0.4]]

#predict class
print(model.predict(new_observation))

#create Gaussian Naive Bayes object with prior probabilities of each class
clf = GaussianNB(priors=[0.25, 0.25, 0.5])

#train model
model = classifier.fit(features, target)

print(model.predict(new_observation))