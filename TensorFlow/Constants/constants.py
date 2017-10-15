import tensorflow as tf

#Constants

#Use add to do a simple addition
a = tf.add(2,3)
with tf.Session() as sess:
    print (sess.run(a))

#Use random_normal to generate a tensor with a mean value and standard deviation
norm = tf.random_normal([2, 3], mean=-1, stddev=4)
with tf.Session() as sess:
    print (sess.run(norm))

#Use random_shuffle to shuffle a given tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)
with tf.Session() as sess:
    print (sess.run(shuff))

#Give the seed argument to random_normal to generate repeatable tensors
norm = tf.random_normal([2, 3], seed=1234)
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))

sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))

#Use tf.fill to fill a matrix with a given number
fill_const = tf.fill([2, 3], 9)
sess = tf.Session()
print(sess.run(fill_const))


#Variables

#initialize and print a simple matrix variable
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)
print (my_int_variable)


#Initialize a variable using another tensor
tensor_variable = tf.get_variable("other_variable", dtype=tf.int32,
  initializer=tf.constant([23, 42]))


# create variable s with scalar value
s = tf.Variable(3, name="scalar")
print (s)


# create variable v as a vector
v = tf.Variable([1, 2], name="vector")
print (v)

# create variable m as a 3x2 matrix
m = tf.Variable([[1, 2], [2, 3], [3, 4]], name="matrix")
print (m)

# create variable t as 500x12 tensor, filled with zeros
t = tf.Variable(tf.zeros([500,12]))
print (t)


# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])

# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)

with tf.Session() as sess:
    # feed [1, 2, 3] to placeholder a via the dict{a: [1, 2, 3]}
    # fetch value of c
    print (sess.run(c, {a: [1, 2, 3]}))


#Operations

#Add
a = tf.subtract(2,3)

#Split
