
#load libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import  make_classification

#Generate features matrix and target vector
X, y = make_classification(n_samples=10000, n_features = 3, n_informative=3, n_redundant=0,
                           n_classes = 2, random_state=1)

#Create logistic regression
logit = LogisticRegression()

#cross - validation model using accuracy
print(cross_val_score(logit, X, y, scoring='accuracy'))

# Cross-validate model using precision
print(cross_val_score(logit, X, y, scoring="precision"))

# Cross-validate model using recall
cross_val_score(logit, X, y, scoring="recall")

# Cross-validate model using f1
cross_val_score(logit, X, y, scoring="f1")


# Load library
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                    test_size=0.1,
                                                    random_state=1)
# Predict values for training target vector
y_hat = logit.fit(X_train, y_train).predict(X_test)

# Calculate accuracy
print(accuracy_score(y_test, y_hat))










