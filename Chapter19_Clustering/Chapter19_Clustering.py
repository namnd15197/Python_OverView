
#load libraries

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#load data
iris = datasets.load_iris()
features = iris.data

#Standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

#create k-mean object
cluster = KMeans(n_clusters=3, random_state=0)

#train model 
model = cluster.fit(features_std)

print(model.labels_)

#create new observation
new_observation = [[0.8, 0.8, 0.8, 0.8]]

#predict observation's cluster
print(model.predict(new_observation))

print(model.cluster_centers_)
