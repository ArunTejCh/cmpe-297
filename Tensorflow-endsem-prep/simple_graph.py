import tensorflow as tf
import tempfile

tf.InteractiveSession()

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = tf.constant(4, dtype=tf.int32, name="c")

d = tf.add_n((a, b, c), name="ADDITION")
graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

print(d.eval())


