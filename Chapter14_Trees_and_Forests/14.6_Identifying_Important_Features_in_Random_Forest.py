
#load libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create random forest classifier object

randomforest = RandomForestClassifier(random_state = 0, n_jobs=-1)

#train model
model_gini = randomforest.fit(features, target)

#calculate feature important
importances = model_gini.feature_importances_

#sort feature importances in descending order
indices =np.argsort(importances)[::-1]

#rearrange feature names so they match the sorted feature importances
names = [iris.feature_names[i] for i in indices]

#create a plot
plt.figure()
#create plot tille
plt.title("Feature Importance")

#add bars
plt.bar(range(features.shape[1]), importances[indices])

#add feature names as x-axis labels
plt.xticks(range(features.shape[1]), names, rotation=90)

#show plot
plt.show()

