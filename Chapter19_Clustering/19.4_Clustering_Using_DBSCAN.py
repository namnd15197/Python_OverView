#load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

#load data
iris = datasets.load_iris()
features = iris.data

#standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

#create meanshift object
cluster = DBSCAN(n_jobs=-1)

#train model
model = cluster.fit(features_std)

print(model.labels_)
