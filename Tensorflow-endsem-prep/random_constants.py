import tensorflow as tf

e = tf.zeros([2, 3], dtype=tf.float64, name=None)
print(e)
print(e.eval())

x = tf.random_normal([2, 3], mean=0.0, stddev=1, dtype=tf.float64, seed=None, name="randX")
print(x)
print(x.eval())

y = tf.random_normal([2, 3], mean=5, stddev=50, dtype=tf.float64, seed=545, name="randY")
print(y)
print(y.eval())

z = tf.random_normal([2, 3], mean=5, stddev=50, dtype=tf.float64, seed=546, name="randZ")
print(z)
print(z.eval())

x = tf.truncated_normal([2, 3], mean=0, stddev=1, dtype=tf.float64, seed=None, name="randX")
print(x)
print(x.eval())

x = tf.random_uniform([2, 3], minval=0, maxval=1, dtype=tf.float64, seed=None, name="randX")
x2 = tf.random_shuffle(x)

with tf.Session() as sess:
    a, b = sess.run([x, x2])
    print(a)
    print(b)

x = tf.random_uniform([2, 3], minval=0, maxval=1, dtype=tf.float64, seed=None, name="randX")
x2 = tf.random_crop(x, [2, 2])

with tf.Session() as sess:
    a, b = sess.run([x, x2])
    print(a)
    print(b)


x = tf.multinomial(tf.log([[10.0, 10.0]]), 6)
print(x)
print(x.eval())

x = tf.random_gamma([10], [5, 50])
print(x)
print(x.eval())
