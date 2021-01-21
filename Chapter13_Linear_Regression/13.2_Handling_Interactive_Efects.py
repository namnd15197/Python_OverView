
#load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

#load data twith only two features
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

#create interaction term
interaction = PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)

#create linear regression
regression = LinearRegression()

#fit the linear regression
model = regression.fit(features_interaction, target)
# View the feature values for first observation
print(features[0])

#import library
import numpy as np
#foreach observation, multiply the values of the first and second feature
interaction_term= np.multiply(features[:,0], features[:, 1])

#view interaction term for first observation
print(interaction_term[0])

# View the values of the first observation
print(features_interaction[0])

