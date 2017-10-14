#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:38:34 2017

@author: DNN - CMPE 297
"""

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X, y = sklearn.datasets.make_moons(100, noise=0.20)
y = y.reshape(len(y), 1)
print (X)
print (y)
# plot the dataset - this is a complete preprocessed(cleaned, normalized, duplicates removed)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

# Splitting into training and testing set
data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("training set ", data_train.shape)


# Sigmoid Activation Function
def sigmoid_activation_function(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid Function - to calculate the slope
def derivatives_sigmoid(x):
    return x * (1 - x)


# Setting the number of epochs aka training iterations
epoch = 5000
# Setting learning rate i.e. how much the weight should be changed to correct the error each time
lr = 0.1
# number of features in data set
inputlayer_neurons = X.shape[1]
print("Number of features in the dataset: ", inputlayer_neurons)
# number of hidden layers neurons
hiddenL_neurons = 3
# number of neurons at output layer
output_neurons = 1

# weight and bias initialization using random function in numpy
wh = np.random.uniform(size=(inputlayer_neurons, hiddenL_neurons))
print("random weight:", wh)
bh = np.random.uniform(size=(1, hiddenL_neurons))
wout = np.random.uniform(size=(hiddenL_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))
output = 0
for i in range(epoch):
    # Forward Propogation
    hiddenL_input = np.dot(X, wh) + bh
    hiddenL_activations = sigmoid_activation_function(hiddenL_input)
    outputL_input = np.dot(hiddenL_activations, wout) + bout
    #output
    output = sigmoid_activation_function(outputL_input)
    # -----------
    # Backpropagation
    E = (y - output)
    outputL_slope = derivatives_sigmoid(output)
    hiddenL_slope = derivatives_sigmoid(hiddenL_activations)
    d_output = E * outputL_slope
    Error_at_hidden_layer = np.dot(d_output, wout.transpose())
    d_hiddenlayer = Error_at_hidden_layer * hiddenL_slope
    wout = wout + np.dot(hiddenL_activations.transpose(), d_output)*lr
    wh = wh + np.dot(X.transpose(), d_hiddenlayer)*lr
    bh = bh + np.sum(d_hiddenlayer, axis=0, keepdims = True) * lr
    bout = bout + np.sum(d_output, axis=0, keepdims = True)*lr

# Plot the output
plt.figure(2)
plt.scatter(output, y)
plt.show()