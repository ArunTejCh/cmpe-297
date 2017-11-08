import tensorflow as tf

tf.InteractiveSession()

a = tf.zeros([2,2])
b = tf.ones([2,2])

print(tf.reduce_sum(b,reduction_indices=1).eval())

print(a.get_shape())

print(tf.reshape(a, (1,4)).eval())

x = tf.add_n([2, 3, 4, 5])

print(x)

print(x.eval())

with tf.Session() as sess:
    print(sess.run(x))

# you explicitly name the nodes in your graph

k = tf.constant(3, name="k")
l = tf.constant(5, name="l")

m = tf.add(k, l, name="add")

print(m.eval())

a = tf.constant([2, 3])
b = tf.constant([4, 5])
print(tf.div(a,b).eval())
print(tf.mod(a,b).eval())
print(a.eval(), b.eval())
a = tf.reshape(a, [1,2])
b = tf.reshape(b, [2,1])
print(a.eval(), b.eval())
c= tf.matmul(a,b)
print(c.eval())

# tf.Variables

scalar = tf.Variable(3, name="scalar")
vector = tf.Variable([1,2], name="vector")
matrix = tf.Variable([[23, 34],[2, 5]], name="matrix")
tensor = tf.Variable(tf.zeros([100, 10]), name="zero_matrix")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(scalar.eval())
    print(vector.eval())
    print(matrix.eval())
    print(tensor.eval())


V = tf.Variable(50)
assign = V.assign(150)


with tf.Session() as sess:
    print(sess.run(V.initializer))
    print(sess.run(V))
#    print(sess.run(assign))


a = tf.placeholder(tf.float32,[3])
b = tf.constant([5,5,5], dtype=tf.float32)

c = a + b

print(c.eval({a: [12,4,6]}))