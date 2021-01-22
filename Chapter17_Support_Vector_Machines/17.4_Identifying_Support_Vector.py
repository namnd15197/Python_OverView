#load libraries
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#load data with only two class
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

#standardize features 
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#create support vector classifier object
svc = SVC(kernel='linear', random_state=0)

#train classifier
model = svc.fit(features_standardized, target)

#view support vectors
print(model.support_vectors_)

print("Support: ", model.support_)
print("_n_support: ", model.n_support_)