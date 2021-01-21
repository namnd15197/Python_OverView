
#Training a Decision Tree Classifier

#load libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

print(features[0])
#create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0)

#train model
model = decisiontree.fit(features, target)

# Make new observation
observation = [[ 5, 4, 3, 2]]

# Predict observation's class
print(model.predict(observation))


# View predicted class probabilities for the three classes
print(model.predict_proba(observation))

#create decision tree classifier object using entropy
decisiontree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)
model_entropy = decisiontree_entropy.fit(features, target)

print(model_entropy.predict(observation))



