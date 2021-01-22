#load libraries
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models, layers

#set random seed
np.random.seed(0)

#set the number of features we want
number_of_features = 10000

#load data and target vector from movie review data
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

#convert moive review data to one-hot encoded feature matrix
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

#start neural network
network = models.Sequential()

#add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))

#add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation="sigmoid"))

#compile neural network
network.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

#train neural network
history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      verbose=1,
                      batch_size=100,
                      validation_data=(features_test, target_test))

print(history)