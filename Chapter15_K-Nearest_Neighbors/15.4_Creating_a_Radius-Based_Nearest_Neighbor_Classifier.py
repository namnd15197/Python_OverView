#load libraries
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create standardizer
standardizer = StandardScaler()

#Standardize feature
features_standardized = standardizer.fit_transform(features)

#Train radius neighbors classifier
rnn = RadiusNeighborsClassifier(radius=0.5, n_jobs=-1).fit(features_standardized, target)

#create two observation
new_observations = [[1, 1, 1, 1]]

#predict the class of two observations
print(rnn.predict(new_observations))
