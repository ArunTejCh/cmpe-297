import tensorflow as tf

tf.InteractiveSession()

t1 = tf.constant([[1,2]])
print(tf.shape(t1).eval())

t2 = tf.constant([[1,2], [3,4]])
print(tf.shape(t2).eval())

m3 = tf.matmul(t1,t2)

print(m3.eval())

m4 = t1*t2

print(m4.eval())


print("reduce_mean: ",tf.reduce_mean([1,4,4], axis=0).eval())
print("reduce_mean_2d: ",tf.reduce_mean([[1.,4.],[3.,4.]]).eval())
print("reduce_mean_2d: ",tf.reduce_mean([[1.,4.],[3.,4.]], axis=1).eval())
print("reduce_sum: ",tf.reduce_sum([1,4,4], axis=0).eval())

x = [[1,2,3], [0,1,5]]

print(tf.argmax(x, axis=1).eval())

print(tf.squeeze([[0], [1], [2]]).eval())

print(tf.expand_dims([0, 1, 2], 1).eval())

print(tf.one_hot([[0], [1], [2], [1]],depth=3).eval())

print(tf.cast([1.2, 2.3, 4.5, 3.2], dtype=tf.int32).eval())

for x,y in zip([1,2,3], [4,8,9]):
    print(x,y)

