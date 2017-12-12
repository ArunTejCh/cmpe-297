import tensorflow as tf


tf.InteractiveSession()
a = tf.constant([1, 2])
b = tf.constant([3, 4])

x1 = tf.concat([a, b], axis=0, name="concat")
x2 = tf.slice(a, [0], [1], name="slice")

with tf.Session() as sess:
    y1, y2 = sess.run([x1, x2])
    print(y1, "\n", y2)

x1 = a.shape
print(x1)
x2 = tf.rank(a)
print(x2.eval())


