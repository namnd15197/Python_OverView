#load libraries

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

#load data
iris = datasets.load_iris()
features = iris.data

#standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

#create meanshift object
cluster = AgglomerativeClustering(n_clusters=3)

#training model
model = cluster.fit(features_std)

#show cluster membership
print(model.labels_)
