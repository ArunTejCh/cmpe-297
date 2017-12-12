# Useful Guidelines to build Neuralnet for image Style transfer
# Please download vggnet from http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
# You can use the below sample parameters to start with. You can modify them too.
# learning rate = 0.001, epochs = 4000, beta1 = 0.9, beta2 = 0.999, original image weight = 5.0, style image weight = 500.0
# VGGNET-19 Layer architecture ['conv1_1', 'relu1_1'],['conv1_2', 'relu1_2', 'pool1'],['conv2_1', 'relu2_1'],['conv2_2', 'relu2_2', 'pool2'],['conv3_1', 'relu3_1'],['conv3_2', 'relu3_2'],['conv3_3', 'relu3_3'],['conv3_4', 'relu3_4', 'pool3'],['conv4_1', 'relu4_1'],['conv4_2', 'relu4_2'],['conv4_3', 'relu4_3'],['conv4_4', 'relu4_4', 'pool4'],['conv5_1', 'relu5_1'],['conv5_2', 'relu5_2'],['conv5_3', 'relu5_3'],['conv5_4', 'relu5_4']

import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
import time
import os

# Save original and style image path into respective variables under the current project dir
img_output_folder = 'output/'
style_image_location = 'images/style_image.jpg'
original_image_location = 'images/content_image.jpg'
vgg19_model_file = 'imagenet-vgg-verydeep-19.mat'
image_width = 458
image_height = 326
mean = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
start = time.time()
print("Starting")

# Hyperparameters
NOISE_RATIO = 0.3
# Constant to put more emphasis on content loss.
alpha = 5
# Constant to put more emphasis on style loss.
beta = 500
learning_rate = 0.1
epochs = 1000

style_layer_weights = [
    ('conv1_1', 4.0),
    ('conv2_1', 3.5),
    ('conv3_1', 3.0),
    ('conv4_1', 1.5),
    ('conv5_1', 0.5),
]


# Read the original and style images using scipy.misc.imread

def load_img(path):
    # Reshape the style image to target image shape
    image = scipy.misc.imresize(scipy.misc.imread(path), [image_height, image_width])
    image = np.reshape(image, ((1,) + image.shape))
    image = image - mean
    return image


content_image = load_img(original_image_location)

style_image = load_img(style_image_location)


# Extract network information
# Step-1 load vgg data in to a matrix using ---scipy.io.loadmat(downloaded imagenet-vgg-verydeep-19.mat)
# Step-2 Compute normalization matrix --- vgg_data_matrix['normalization'][0][0][0]
# Step-3 Compute the mean
# Step-4 Extract network weights --- vgg_data_matrix['layers'][0]
# Create VGG-19 Network using the VGGNET-19 Layer architecture mentioned above
# Iterate over each layer and load respective layer paramenters (weights, biases) for conv, relu and pool
def load_model(path):
    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        W = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(
            prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    model = {}
    model['input'] = tf.Variable(np.zeros((1, image_height, image_width, 3)), dtype='float32')
    model['conv1_1'] = _conv2d_relu(model['input'], 0, 'conv1_1')
    model['conv1_2'] = _conv2d_relu(model['conv1_1'], 2, 'conv1_2')
    model['avgpool1'] = _avgpool(model['conv1_2'])
    model['conv2_1'] = _conv2d_relu(model['avgpool1'], 5, 'conv2_1')
    model['conv2_2'] = _conv2d_relu(model['conv2_1'], 7, 'conv2_2')
    model['avgpool2'] = _avgpool(model['conv2_2'])
    model['conv3_1'] = _conv2d_relu(model['avgpool2'], 10, 'conv3_1')
    model['conv3_2'] = _conv2d_relu(model['conv3_1'], 12, 'conv3_2')
    model['conv3_3'] = _conv2d_relu(model['conv3_2'], 14, 'conv3_3')
    model['conv3_4'] = _conv2d_relu(model['conv3_3'], 16, 'conv3_4')
    model['avgpool3'] = _avgpool(model['conv3_4'])
    model['conv4_1'] = _conv2d_relu(model['avgpool3'], 19, 'conv4_1')
    model['conv4_2'] = _conv2d_relu(model['conv4_1'], 21, 'conv4_2')
    model['conv4_3'] = _conv2d_relu(model['conv4_2'], 23, 'conv4_3')
    model['conv4_4'] = _conv2d_relu(model['conv4_3'], 25, 'conv4_4')
    model['avgpool4'] = _avgpool(model['conv4_4'])
    model['conv5_1'] = _conv2d_relu(model['avgpool4'], 28, 'conv5_1')
    model['conv5_2'] = _conv2d_relu(model['conv5_1'], 30, 'conv5_2')
    model['conv5_3'] = _conv2d_relu(model['conv5_2'], 32, 'conv5_3')
    model['conv5_4'] = _conv2d_relu(model['conv5_3'], 34, 'conv5_4')
    model['avgpool5'] = _avgpool(model['conv5_4'])
    return model


model = load_model(vgg19_model_file)


# Apply 'relu4_2' to original image
def content_loss_func(sess, model):
    def _content_loss(p, x):
        N = p.shape[3]
        M = p.shape[1] * p.shape[2]
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))

    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


# 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1' to style image
# Get network parameters by extracting the network information
# Get network parameters
# Form the style image network and form gram Matrix for style layers
def style_loss_func(sess, model):
    def _gram_matrix(F, N, M):
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]
        A = _gram_matrix(a, N, M)
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in style_layer_weights]
    W = [w for _, w in style_layer_weights]
    loss = sum([W[l] * E[l] for l in range(len(style_layer_weights))])
    return loss


# Make the Combined Image
def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
    noise_image = np.random.uniform(
        -20, 20,
        (1, image_height, image_width, 3)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


input_image = generate_noise_image(content_image)
sess.run(tf.initialize_all_variables())
sess.run(model['input'].assign(input_image))
# Construct content_loss using content_image.
# Calculate the content loss
sess.run(model['input'].assign(content_image))
content_loss = content_loss_func(sess, model)

# Calculate style loss from Style Image
sess.run(model['input'].assign(style_image))
style_loss = style_loss_func(sess, model)

# Calculate the combined loss (content loss + style loss)
total_loss = alpha * content_loss + beta * style_loss

# Declare Optimizer and minimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(total_loss)


def save_image(path, image):
    image = image + mean
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


# Initialize all Variables using session and start Training
sess.run(tf.initialize_all_variables())
sess.run(model['input'].assign(input_image))
for it in range(epochs):
    sess.run(train_step)
    if it % 100 == 0:
        # Print every 100 iteration.
        combined_image = sess.run(model['input'])
        print('Iteration %d' % (it))
        print('sum : ', sess.run(tf.reduce_sum(combined_image)))
        print('cost: ', sess.run(total_loss))
        cur = time.time()
        print("Time elapsed: ", round((cur - start) / 60, 2), " mins")
        if not os.path.exists(img_output_folder):
            os.mkdir(img_output_folder)

        filename = 'output/%d.png' % (it)
        save_image(filename, combined_image)

# You can use ---scipy.misc.imsave() for saving the image
save_image('output/art.jpg', combined_image)
end = time.time()
print("Time elapsed: ", round((end - start) / 60, 2), " mins")
