#load data
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Brazil is best',
                      'Germany beats both'])

#create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

#create feature matrix
features = bag_of_words.toarray()
print(features);

#create target vector
target = np.array([0, 0, 1])

#create multinomial naive Bayes object with prior probabilities of each class
classifier = MultinomialNB(class_prior =[0.25, 0.5])

#train model
model = classifier.fit(features, target)

# Create new observation
new_observation = [[0, 0, 0, 1, 0, 1, 0]]

# Predict new observation's class
print(model.predict(new_observation))