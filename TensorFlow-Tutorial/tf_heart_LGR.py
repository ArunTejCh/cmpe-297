import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile

tf.reset_default_graph()
data = pd.read_csv('heart.csv')

dummy_data = pd.get_dummies(data['famhist'], prefix='famhist', drop_first=False)
data = pd.concat([data, dummy_data], axis=1)

data = data.drop(['famhist'], axis=1)

cols = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']

labels = data['chd']
# min-max scaling
for each in cols:
    data[each] = (data[each] - data[each].min()) / (data[each].max() - data[each].min())

features = data.drop(['chd'], axis=1)
features, labels = np.array(features), np.array(labels)


# Define fraction to split into training and testing parts
split_frac = 0.85
n_records = len(features)
split_idx = int(split_frac*n_records)

train_X, train_Y = features[:split_idx], labels[:split_idx]
test_X, test_Y = features[split_idx:], labels[split_idx:]

n_labels= 2
n_features = 10
learning_rate = 0.1
n_epochs= 300
n_hidden1 = 10

inputs = tf.placeholder(tf.float32, [None, 10], name='inputs')
labels = tf.placeholder(tf.int32, [None, ], name='output')
labels_one_hot = tf.one_hot(labels, 2)

weights = {
        'hidden_layer': tf.Variable(tf.truncated_normal([n_features, n_hidden1], stddev=0.1)),
        'output': tf.Variable(tf.truncated_normal([n_hidden1, n_labels], stddev=0.1))
    }

bias = {
        'hidden_layer': tf.Variable(tf.zeros([n_hidden1])),
        'output': tf.Variable(tf.zeros(n_labels))
    }

hidden_layer = tf.nn.bias_add(tf.matmul(inputs, weights['hidden_layer']), bias['hidden_layer'])
hidden_layer = tf.nn.relu(hidden_layer)
logits = tf.nn.bias_add(tf.matmul(hidden_layer, weights['output']), bias['output'])

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot)
cost = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
# create a summary for our cost and accuracy
# testing the model on test data
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                inputs: train_X, labels: train_Y})
            print('step %d, training accuracy %g' % (epoch, train_accuracy))
        _, loss = sess.run([optimizer, cost], feed_dict={inputs: train_X, labels: train_Y})

    print("Out of sample accuracy:", sess.run(accuracy, ({inputs: test_X, labels: test_Y})))
