
#17.1 Training a Linear Classifier

#load libraries
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#load data with only two classes and two features
iris = datasets.load_iris()
features = iris.data[:100, :2]
target = iris.target[:100]

#standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#create support vector machine
svc = LinearSVC(C=1.0)

#Train model
model = svc.fit(features_standardized, target)

#create bew observation
new_observation =[[-2, 3]]

#predict class of new observation
print(svc.predict(new_observation))

#load library
from matplotlib import pyplot as plt

#plot data points and color using their class
color = ["black" if c == 0 else "lightgrey" for c in target]
plt.scatter(features_standardized[:,0], features_standardized[:,1], c= color)

#create the hyperplane
w = svc.coef_[0]
a = -w[0] / w[1]
xx =np.linspace(-2.5, 2.5)
yy = a*xx - (svc.intercept_[0]) / w[1]

#plot the hyperplane
plt.plot(xx, yy)
plt.axis("off"), plt.show()