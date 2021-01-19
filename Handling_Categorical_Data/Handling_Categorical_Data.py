
#5.1 Encoding Nominal Categorical Features
# Import libraries
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
# Create feature
feature = np.array([["Texas"],
 ["California"],
 ["Texas"],
 ["Delaware"],
 ["Texas"]])
# Create one-hot encoder
one_hot = LabelBinarizer()
# One-hot encode feature
print("Matrix Encoding: \n", one_hot.fit_transform(feature))

# View feature classes
print("Feature classes: \n", one_hot.classes_)

# Reverse one-hot encoding
print("Reverse one-hot encoding: \n", one_hot.inverse_transform(one_hot.transform(feature)))

# Import library
import pandas as pd
# Create dummy variables from feature
print("Matrix encoding with pandas: \n", pd.get_dummies(feature[:,0]))

# Create multiclass feature
multiclass_feature = [("Texas", "Florida"),
                     ("California", "Alabama"),
                     ("Texas", "Florida"),
                     ("Delware", "Florida"),
                     ("Texas", "Alabama")]
# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()
# One-hot encode multiclass feature
print("Multi label binarizer: \n", one_hot_multiclass.fit_transform(multiclass_feature))
print("Feature classes: \n", one_hot_multiclass.classes_)

print("\n")
print("\n")
print("\n")

#5.2 Encoding Ordinal Categorical Features

# Load library
import pandas as pd
# Create features
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})

# Create mapper
scale_mapper = { "Low":1,
                 "Medium":2,
                 "High":3}
# Replace feature values with scale
dataframe["Score"].replace(scale_mapper)
print("\n")
print("\n")
print("\n")

#5.3 Encoding Dictionaries of Features

# Import library
from sklearn.feature_extraction import DictVectorizer
# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
 {"Red": 4, "Blue": 3},
 {"Red": 1, "Yellow": 2},
 {"Red": 2, "Yellow": 2}]
# Create dictionary vectorizer
dictvectorizer = DictVectorizer(sparse=False)
# Convert dictionary to feature matrix
features = dictvectorizer.fit_transform(data_dict)
# View feature matrix
print("Feature matrix: \n", features)

# Get feature names
feature_names = dictvectorizer.get_feature_names()
# View feature names
print("feature name: \n", feature_names)

# Import library
import pandas as pd
# Create dataframe from features
print(pd.DataFrame(features, columns=feature_names))

print("\n")
print("\n")
print("\n")


#5.4 Imputing Missing Class Values

#load library
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45],
             [1, 1.18, 1.33],
             [0, 1.22, 1.27],
             [1, -0.21, -1.19]])
# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array( [[np.nan, 0.87, 1.31],
                        [np.nan, -0.67, -0.22]])

# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])
# Predict missing values' class
imputed_values = trained_model.predict(X_with_nan[:,1:])
# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))
# Join two feature matrices
print("Features: \n", np.vstack((X_with_imputed, X)))

from sklearn.impute import SimpleImputer
# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))
imputer = SimpleImputer(strategy='most_frequent')
print("feature: \n", imputer.fit_transform(X_complete))


#5.5 Handling Imbalanced Classes

# Load libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load iris data
iris = load_iris()
# Create feature matrix
features = iris.data
# Create target vector
target = iris.target
# Remove first 40 observations
features = features[40:,:]
target = target[40:]

# Create binary target vector indicating if class 0
target = np.where((target == 0), 0, 1)
# Look at the imbalanced target vector
target

print("Targer: \n", target)

# Create weights
weights = {0: .9, 1: 0.1}
# Create random forest classifier with weights
RandomForestClassifier(class_weight=weights)
RandomForestClassifier(bootstrap=True, class_weight={0: 0.9, 1: 0.1},
                         criterion='gini', max_depth=None, max_features='auto',
                         max_leaf_nodes=None, min_impurity_decrease=0.0,
                         min_impurity_split=None, min_samples_leaf=1,
                         min_samples_split=2, min_weight_fraction_leaf=0.0,
                         n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                         verbose=0, warm_start=False)



# Indicies of each class' observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

# Number of observations in each class
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# For every observation of class 0, randomly sample
# from class 1 without replacement
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)

# Join together class 0's target vector with the
# downsampled class 1's target vector
print("Join target vector: ", np.hstack((target[i_class0], target[i_class1_downsampled])))

# Join together class 0's feature matrix with the
# downsampled class 1's feature matrix
print("Features join: \n", np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5])

# For every observation in class 1, randomly sample from class 0 with replacement
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
# Join together class 0's upsampled target vector with class 1's target vector
print("Join Features: \n", np.concatenate((target[i_class0_upsampled], target[i_class1])))

# Join together class 0's upsampled feature matrix with class 1's feature matrix
print("joint feature: \n", np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5])









