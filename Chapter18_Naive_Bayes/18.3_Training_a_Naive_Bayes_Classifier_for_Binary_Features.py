#load libraries
import numpy as np
from sklearn.naive_bayes import BernoulliNB

#Create three binary features
features = np.random.randint(2, size=(100, 3))

#create a binary target vector
target = np.random.randint(2, size=(100, 1)).ravel()

#create Bernoulli Naive Bayes object with prior probabilities of each class
classifier = BernoulliNB(class_prior=[0.25, 0.5])

#train model
model = classifier.fit(features, target)

model_uniform_prior = BernoulliNB(class_prior=None, fit_prior=True)