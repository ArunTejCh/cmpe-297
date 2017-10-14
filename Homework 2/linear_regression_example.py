import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas


# Phase 1: Assemble the graph
# Step 1: read in data in to read_data from LinearRegression.csv file
data = pandas.read_csv('LinearRegression.csv')

n_samples = data.shape[0]# calculate numberLinearRegression.csv of samples
print (n_samples)
# Step 2:
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Step 3:
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Step 4:
Y_predicted = X * w + b

# Step 5:
loss = tf.square(Y_predicted - Y)

# Step 6:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
 
# Phase 2: Train the model
with tf.Session() as sess:
	# Step 7: 
    sess.run(tf.global_variables_initializer())
    # Step 8: train the model
    # Step 8: train the model
    for i in range(50):
        total_loss = 0
        for index, d in data.iterrows():
            # Run the optimizer and loss, store the loss value as l
            print (d['X'])
            print (d['Y'])
            yp, l, _ = sess.run([Y_predicted, loss, optimizer], feed_dict={X: [d['X']], Y: [d['Y']]})
            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss / n_samples))

    w, b = sess.run([w, b])
print (w)
print (b)

# plot the results
X, Y = data[data.columns[0]], data[data.columns[1]]
plt.plot(X, Y, 'bo', label='Actual data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()
