import tensorflow as tf

a = tf.Variable(tf.constant([[1, 2], [3, 4]]))
print(a)
b = tf.assign(a, [[4, 5], [6, 7]])
print(b)
with tf.Session() as sess:
    print(sess.run(a.initializer))
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(a))


b = tf.assign_add(a, [[4, 5], [6, 7]])
print(b)
with tf.Session() as sess:
    print(sess.run(a.initializer))
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(a))


b = tf.assign_sub(a, [[4, 5], [6, 7]])
print(b)
with tf.Session() as sess:
    print(sess.run(a.initializer))
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(a))