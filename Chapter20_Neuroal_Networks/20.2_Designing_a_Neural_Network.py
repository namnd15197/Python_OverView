
#load libraries
from keras import models
from keras import layers

#start neural network
network = models.Sequential()

#add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu", input_shape=(10, 1)))

#add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation="relu"))

#add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation="sigmoid"))

#compile neural network
network.compile(loss="binary_crossentropy", #cross-entropy
                optimizer="rmsprop", #root mean square propagation
                metrics=["accuracy"]
                )
