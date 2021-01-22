
#load libraries
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#load data with only two classes
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]

#make class highly imbalanced by removing first 40 observations
features = features[40:,:]
target = target[40:]

#create target vector indicating if class 0 otherwise 1
target = np.where((target==0), 0, 1)

#standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#create support vector classifier
svc = SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=0)

#train classifier
model = svc.fit(features_standardized, target)