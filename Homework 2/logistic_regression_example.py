import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

n_epochs = 20
batch_size = 128
learning_rate = 0.5

# Step 1: Read in data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True) 
nb_classes = 10
# Step 2
X = tf.placeholder(tf.float32, [batch_size,784])
Y = tf.placeholder(tf.float32, [batch_size,nb_classes])

# Step 3
w = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Step 4
logits = tf.matmul(X, w) + b

# Step 5:
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)

# Step 6:
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs): 
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			#print (X_batch)
			#print (Y_batch)
			# Run the optimizer and loss, store the loss value as loss_batch
			loss_batch, _ = sess.run([loss, optimizer], feed_dict={X: X_batch, Y: Y_batch})
			 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))


	# test the model
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) 
	
	n_batches = int(mnist.test.num_examples/batch_size)
	#print (n_batches)
	total_correct_preds = 0
	
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}) 
		#print (accuracy_batch)
		total_correct_preds += sum(accuracy_batch)
		#print (total_correct_preds)
	
	print('Accuracy {0}'.format((total_correct_preds/mnist.test.num_examples)*100))
