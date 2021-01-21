
#load data
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create random forest classifier
randomforest = RandomForestClassifier(random_state = 0, n_jobs=-1)

#create object that selects features with importance greater 
#than or equal to a threshold
selector = SelectFromModel(randomforest, threshold=0.3)

#feature new feature matrix using selector
features_important = selector.fit_transform(features, target)

#train random forest using most important features
model = randomforest.fit(features_important, target)

observation = [[0.02, 16]]
print(model.predict(observation))
