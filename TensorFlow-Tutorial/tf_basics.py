# TensorFlow basics tutorial

import tensorflow as tf

node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(7.0)

print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node4 = tf.add(node3, tf.add(node1, node2))
node5 = node3 + node2 + node1

print("node4: ", node4)
print(sess.run(node4))

print("node5: ", node5)
print(sess.run(node5))

a = tf.placeholder(tf.float32)  # Used when you don't know the shape of a or the shape can change later
b = tf.placeholder(tf.float32)
c = a + b # works the same as tf.add(a,b)

print(a, b, c)

print(sess.run(c, {a: 4, b: 5}))  # A is a rank 0 scalar with shape []

print(sess.run(c, feed_dict= {a: [4, 3], b: [5, 45]}))  # A is a rank 1 tensor with shape [2]

add_and_triple = c * 3

print(sess.run(add_and_triple,feed_dict={a: [7, 2], b: [5, 45]}))

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(dtype=tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)  # Learning rate is the argument passed
train = optimizer.minimize(loss)

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

sess.run(init)

for i in range(1000):
    sess.run([train, loss], feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# Lets see the results
weight_W, weight_b, final_loss = sess.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(weight_W, weight_b, final_loss)

