#load libraries
from sklearn.tree import DecisionTreeRegressor #giong voi DecisionTreeClassifier nhung k dung criterion la 'gini' va 'entropy'
from sklearn import datasets

#load data with only two features
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target

#create decision tree classifier object
decisiontree = DecisionTreeRegressor(random_state = 0)

#train model
model = decisiontree.fit(features, target)

# Make new observation
observation = [[0.02, 16]]

# Predict observation's value
print("MSE: ", model.predict(observation))

# Create decision tree classifier object using entropy
decisiontree_mae = DecisionTreeRegressor(criterion="mae", random_state=0)
# Train model
model_mae = decisiontree_mae.fit(features, target)

print("MAE: ", model_mae.predict(observation))