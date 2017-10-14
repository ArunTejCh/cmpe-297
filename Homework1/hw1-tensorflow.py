import tensorflow as tf
import random
import pandas
sess = tf.InteractiveSession()

#Question 1
a = tf.constant(random.uniform(-1, 1))
b = tf.constant(random.uniform(-1, 1))
print (a.eval())
print (b.eval())

f1 = lambda: tf.add(a,b)
f2 = lambda: 0.0
r = tf.case([(tf.less(a, b), f1)], default=f2)

print (r.eval())

#Question 2
norm = tf.random_normal([16, 16], mean=-1, stddev=4)
print (norm.eval())
det = tf.matrix_determinant(norm)
print (det.eval())

#Question 3

#part a
ten3 = tf.constant([30.05088806, 17.61298943, 41.19073486, 19.35532951,
31.97266006, 16.67541885, 28.08450317, 21.74983215,
32.94445419, 30.45999146, 39.06485367, 32.01657104,
26.88236427, 27.56035233, 10.20379066, 22.51215172,
30.71149445, 24.59134293, 56.05556488, 30.66994858])

where = tf.greater(ten3, 25)
print (where.eval())
indices = tf.where(where)
print (indices.eval())

#part b
where = tf.greater(ten3, 30)
print (where.eval())
indices = tf.where(where)
print (indices.eval())
extract=tf.gather(ten3,indices)
print (extract.eval())

#Question 4
ten4 = tf.constant([[-1, 0, 2], [1, 0, 2]])
print (ten4.eval())
ten4zeros = tf.zeros_like(ten4)
print (ten4zeros.eval())

compare = tf.equal(ten4,ten4zeros)
print (compare.eval())


#Question 5
df = pandas.read_csv('place_holder.csv')
print (df)
mat = df.as_matrix()
print (mat)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
C = tf.ones_like(X)

# Define sum operation
sum = tf.add(tf.add(X,Y),C)

print(sess.run(sum,feed_dict={X:df[df.columns[0]]
                                   ,Y:df[df.columns[1]]}))




