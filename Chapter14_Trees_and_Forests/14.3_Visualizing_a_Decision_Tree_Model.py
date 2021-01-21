
#load libraries
import pydotplus as pydot
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0)

#train model
model = decisiontree.fit(features, target)

#create DOT data
dot_data = tree.export_graphviz(decisiontree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)

#draw graph
graph = pydot.graph_from_dot_data(dot_data)

#show graph
Image(graph.create_png())
