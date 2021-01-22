
#load libraries
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#set randomization seed
np.random.seed(0)

#generate two features
features = np.random.randn(200 , 2)

#use a XOR gate (you don't need to know what this is) to generate
#linearly inseparable classes
target_xor = np.logical_xor(features[:,0] > 0, features[:, 1]>0)
target = np.where(target_xor, 0, 1)

#create a support vector machine with a radial basis function kernel
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)

#train the classifier
model = svc.fit(features, target)

#plot observations and decision boundary hyperplane
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier):
    cmap = ListedColormap(("red", "blue"))
    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap = cmap)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker="+", label=cl)

# Create support vector classifier with a linear kernel
svc_linear = SVC(kernel="linear", random_state=0, C=1)

#train model 
print(svc_linear.fit(features, target))

plot_decision_regions(features, target, classifier=svc_linear)
plt.axis("off"), plt.show()


#Create a support vector machine with a radial basis function kernel
svc = SVC(random_state=0, gamma=1, C=1.0)

#train the classifier
model = svc.fit(features, target)

#plot observations and hyperplane
plot_decision_regions(features, target, classifier=svc)
plt.axis("off"), plt.show();