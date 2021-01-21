
#12.1 Selecting Best Models Using Exhaustive Search

import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create logistic regression
logistic = linear_model.LogisticRegression()

#create range of candiate penalty hyperparameter values
penalty =['l1', 'l2']

#create range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)

#create dictionary hyperparameter candidates
hyperparameters = dict(C=C, penalty = penalty)

#create grid search
gridSearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

#fit grid search
best_model = gridSearch.fit(features, target)
#print(best_model)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# Predict target vector
best_model.predict(features)


























































































































