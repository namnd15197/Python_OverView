
#load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

#train model
model = randomforest.fit(features, target)

# Make new observation
observation = [[ 5, 4, 3, 2]]

#predict observation's class
print('Gini: ',model.predict(observation))

# Create random forest classifier object using entropy
randomforest_entropy = RandomForestClassifier(
 criterion="entropy", random_state=0)
# Train model
model_entropy = randomforest_entropy.fit(features, target)

print('entropy: ', model_entropy.predict(observation))
