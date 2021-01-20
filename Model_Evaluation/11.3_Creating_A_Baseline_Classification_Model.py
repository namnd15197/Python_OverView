
#load libraries
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

#load data
iris = load_iris()

#create target vector and feature matrix
features, target = iris.data, iris.target

#split into training and test set
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)

#create dummy classifier
dummy = DummyClassifier(strategy='uniform', random_state=1)

#train model
dummy.fit(features_train, target_train)

#get accuracy score
print("dummy score: \n", dummy.score(features_test, target_test))


#load library
from sklearn.ensemble import RandomForestClassifier

#create classifier
classifier = RandomForestClassifier()

#train model
classifier.fit(features_train, target_train)

#get accuracy score
print("Random Forest Classifier score: \n", classifier.score(features_test, target_test))