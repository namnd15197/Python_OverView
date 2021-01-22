#load libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create standardizer
standardizer = StandardScaler()

#standardize feature
features_standardized = standardizer.fit_transform(features)

#create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

#create a pipe line
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])

#create space of candidate values
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

#create a grid search
classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(features_standardized, target)

# Best neighborhood size (k)
print(classifier.best_estimator_.get_params()["knn__n_neighbors"])