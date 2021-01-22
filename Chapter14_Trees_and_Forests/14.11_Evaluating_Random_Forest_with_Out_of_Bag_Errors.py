
#load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create random forest classifier object
randomforest = RandomForestClassifier( random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)

#train model
model = randomforest.fit(features, target)

#view out of bag error
print("Out of Bag Error: ", randomforest.oob_score_)