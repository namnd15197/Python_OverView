
#16.1 Training a binary Classifier

#load libraries
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#load data with only two classes
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

#standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#create LogisticRegression

logisticregression = LogisticRegression(random_state=0)

#train model
model = logisticregression.fit(features_standardized, target)


# Create new observation
new_observation = [[0.000001, 0.000001, -0.15, 0.000001]]
# Predict class
print(model.predict(new_observation))

# View predicted probabilities
print(model.predict_proba(new_observation))
#array([[ 0.18823041, 0.81176959]])

