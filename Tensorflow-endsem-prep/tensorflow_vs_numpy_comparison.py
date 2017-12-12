import tensorflow as tf
import numpy as np

a = np.zeros((2, 2, 2))
print(a)
print(a.shape)

b = np.ones((4, 3))
sum_1 = np.sum(b, axis=0)
sum_2 = np.sum(b, axis=1)
print(b.shape)
c = np.reshape(b, (3, 4))
print(c.shape)

tf.InteractiveSession();

a = tf.zeros((2, 2, 2))
print(a)
print(a.shape)

b = tf.ones((4, 3))
sum_1_tf = tf.reduce_sum(b, reduction_indices=0).eval()
sum_2_tf = tf.reduce_sum(b, reduction_indices=1).eval()
print(b.shape)
c = tf.reshape(b, (3, 4)).eval()
print(b.shape)
print(c)

a = tf.add(2, 3)
print(a)
print(a.eval())

print("End of the line...")

