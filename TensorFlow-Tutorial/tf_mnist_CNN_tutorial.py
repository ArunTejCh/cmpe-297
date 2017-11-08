import tensorflow as tf
import tempfile
import math
import time

start = time.time()
print("Starting")

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print()
#no_of_batches=int((tf.shape(mnist.test.images)[0])/50)
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")

with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([784, 10]))

with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]))

with tf.name_scope("softmax"):
    y = tf.nn.softmax(tf.matmul(x,W) + b)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_c1 = weight_variable([4, 4, 1, 32])
b_c1 = bias_variable([32])

x_img = tf.reshape(x, [-1, 28, 28, 1])

h_c1 = tf.nn.relu(conv2d(x_img, W_c1) + b_c1)
h_p1 = max_pool_2x2(h_c1)

W_c2 = weight_variable([4, 4, 32, 64])
b_c2 = bias_variable([64])

h_c2 = tf.nn.relu(conv2d(h_p1, W_c2) + b_c2)
h_p2 = max_pool_2x2(h_c2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_p2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('cross_entropy'):
    # this is our cost
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(2e-4).minimize(entropy)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a summary for our cost and accuracy
tf.summary.scalar("cost", entropy)
tf.summary.scalar("accuracy", accuracy)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

summary_op = tf.summary.merge_all()

batch_size=50
no_of_batches=int(mnist.test.images.shape[0]/batch_size)
test_accuracy=0

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(mnist.test.images.shape[0])
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        _, summary = sess.run([train, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        # write log
        train_writer.add_summary(summary, i)

    for i in range(no_of_batches):
        batch = mnist.test.next_batch(batch_size)
        print("current test accuracy is: ", test_accuracy)
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})

print("got accuracy as: ", test_accuracy / no_of_batches)
end = time.time()
print("Time elapsed: ",round((end - start)/60, 2), " mins")