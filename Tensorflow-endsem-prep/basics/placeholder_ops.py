import tensorflow as tf

a = tf.placeholder(dtype=tf.int32, shape=[2])
b = tf.constant([3, 4])

c = a + b

with tf.Session() as sess:
    print(sess.run(c, {a: [1, 2]}))

# In y = mx + c, tf.Variables are used for m,c and placeholders are used for x, y
# Its a bit unintuitive for programmers not used to 2-Phase programming done in tf
# But rest assured there are good reasons for this design


