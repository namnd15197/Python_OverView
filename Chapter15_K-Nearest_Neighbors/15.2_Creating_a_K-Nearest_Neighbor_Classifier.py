#load libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

#load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

#create standardizer
standardizer = StandardScaler()

#standardize features
X_std = standardizer.fit_transform(X)

#Train a knn classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)

#create two observations
new_observations =[[0.75, 0.75, 0.75, 0.75],
                   [1, 1, 1, 1]]

#predict the class of two observations
print(knn.predict(new_observations))

# View probability each observation is one of three classes
print(knn.predict_proba(new_observations))


