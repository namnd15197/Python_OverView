
#load libraries

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state = 0,
                                      max_depth=None,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0)
model = decisiontree.fit(features, target)

observation = [[5, 4, 3, 2]]

print(model.predict(observation))