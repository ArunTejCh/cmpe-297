import tensorflow as tf

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[5., 6.], [7., 8.]])

x1 = tf.matmul(a, b)
x2 = tf.matrix_determinant(a)
x3 = tf.matrix_inverse(a)

with tf.Session() as sess:
    y1, y2, y3 = sess.run([x1, x2, x3])
    print(y1)
    print(y2)
    print(y3)

