import tensorflow as tf


a = tf.constant([1, 2])
b = tf.constant([3, 4])
x1 = tf.add(a, b)
x2 = tf.subtract(b, a)
x3 = tf.multiply(b, a)

with tf.Session() as sess:
    y1, y2, y3 = sess.run([x1, x2, x3])
    print(y1, "\n", y2, "\n", y3)

a = tf.constant([1., 2.])
b = tf.constant([3., 4.])
x1 = tf.div(b, a)
x2 = tf.exp(b, "exp")
x3 = tf.log(b, "log")


with tf.Session() as sess:
    y1, y2, y3 = sess.run([x1, x2, x3])
    print(y1, "\n", y2, "\n", y3)

a = tf.constant([1, 2])
b = tf.constant([3, 4])
x1 = tf.greater(a, b)
x2 = tf.less(a, b)
x3 = tf.equal(b, a)

with tf.Session() as sess:
    y1, y2, y3 = sess.run([x1, x2, x3])
    print(y1, "\n", y2, "\n", y3)

