import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler

google_stocks_train = pd.read_csv('Google_Stock_Price_Train.csv')
google_stocks_train.head()
google_stocks_test = pd.read_csv('Google_Stock_Price_Test.csv')
google_stocks_test.head()

data_to_train = google_stocks_train['Open'].values
data_to_test = google_stocks_test['Open'].values
print('Total number of days in the training dataset: {}'.format(len(data_to_train)))
print('Total number of days in the testing dataset: {}'.format(len(data_to_test)))

scaler = StandardScaler()

scaled_train_dataset = scaler.fit_transform(data_to_train.reshape(-1, 1))
scaled_test_dataset = scaler.fit_transform(data_to_test.reshape(-1, 1))

plt.figure(figsize=(12, 7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('Scaled Google stocks from Jan 2012 to Dec 2016')
plt.xlabel('Days')
plt.ylabel('Scaled value of stocks')
plt.plot(scaled_train_dataset, label='Stocks data')
plt.legend()
#plt.show()

plt.figure(figsize=(12, 7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('Scaled Google stocks from Jan 2017')
plt.xlabel('Days')
plt.ylabel('Scaled value of stocks')
plt.plot(scaled_test_dataset, label='Stocks data')
plt.legend()
#plt.show()


def transform_data(data, window_size):
    X = []
    y = []

    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

        i += 1
    assert len(X) == len(y)
    return X, y


epochs = 200
batch_size = 1

X, y = transform_data(scaled_train_dataset, batch_size)

X_t, y_t = transform_data(scaled_test_dataset, batch_size)

X_train = np.array(X)
y_train = np.array(y)
X_test = np.array(X_t)

print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))

def input_train():
    return ({"data": tf.constant(X_train)}, tf.constant(y_train))

def input_test():
    return ({"data": tf.constant(X_test)})

feature_columns = [
tf.feature_column.numeric_column(key="data")
]

# Train the dataset with tf.estimator
tf_model_ANN = tf.estimator.DNNRegressor(hidden_units=[20, 20], feature_columns=feature_columns)
tf_model_ANN.train(input_fn=input_train, steps=epochs)

# Use the trained model to predict output
out_gen = tf_model_ANN.predict(input_fn=input_test)
y_ = list(itertools.islice(out_gen, X_test.shape[0]))
o = np.array(list(map(lambda x: x['predictions'][0], y_)))

# Inverse transform to get back original stock values
output = scaler.inverse_transform(o)

plt.figure(figsize=(16, 7))
plt.plot(data_to_test.reshape(-1, 1), label='Real Google stock price')
plt.plot(output[1:], label='Predicted Google Stock price')
plt.xlabel('Time in Days')
plt.ylabel('Value of stocks')
plt.legend()
plt.show()