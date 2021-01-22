#load libraries
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#create support vector machine object
svc = SVC(kernel='linear', probability=True, random_state=0)

#train classifier
model = svc.fit(features_standardized, target)

#create new observation
new_observation = [[0.4, 0.4, 0.4, 0.4]]

#view predict probabilities
print(model.predict_proba(new_observation))

#def plot_decision_regions(X, y, classifier):
#    cmap = ListedColormap(("red", "blue"))
#    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
#    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap = cmap)

#    for idx, cl in enumerate(np.unique(y)):
#        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker="+", label=cl)

#plot_decision_regions(features, target, svc)
#plt.axis("off"), plt.show();