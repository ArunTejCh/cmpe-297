#Useful Guidelines to build Neuralnet for image Style transfer
#Please download vggnet from http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
#You can use the below sample parameters to start with. You can modify them too.
#learning rate = 0.001, epochs = 4000, beta1 = 0.9, beta2 = 0.999, original image weight = 5.0, style image weight = 500.0
# VGGNET-19 Layer architecture ['conv1_1', 'relu1_1'],['conv1_2', 'relu1_2', 'pool1'],['conv2_1', 'relu2_1'],['conv2_2', 'relu2_2', 'pool2'],['conv3_1', 'relu3_1'],['conv3_2', 'relu3_2'],['conv3_3', 'relu3_3'],['conv3_4', 'relu3_4', 'pool3'],['conv4_1', 'relu4_1'],['conv4_2', 'relu4_2'],['conv4_3', 'relu4_3'],['conv4_4', 'relu4_4', 'pool4'],['conv5_1', 'relu5_1'],['conv5_2', 'relu5_2'],['conv5_3', 'relu5_3'],['conv5_4', 'relu5_4']


import tensorflow as tf
import scipy.misc
import numpy as np
import scipy.io
import scipy.misc

# Save original and style image path into respective variables under the current project dir
original_image_Location = ''
style_image_Location = ''

# Read the original and style images using scipy.misc.imread


# Reshape the style image to target image shape


# Extract network information
## Step-1 load vgg data in to a matrix using ---scipy.io.loadmat(downloaded imagenet-vgg-verydeep-19.mat)
## Step-2 Compute normalization matrix --- vgg_data_matrix['normalization'][0][0][0]
## Step-3 Compute the mean
## Step-4 Extract network weights --- vgg_data_matrix['layers'][0]


# Create VGG-19 Network using the VGGNET-19 Layer architecture mentioned above
##Iterate over each layer and load respective layer paramenters (weights, biases) for conv, relu and pool 



# Apply 'relu4_2' to original image and 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1' to style image


# Get network parameters by extracting the network information


# Get network parameters


# Form the style image network and form gram Matrix for style layers


# Make the Combined Image


# Calculate the content loss

                
# Claculate style loss from Style Image

    
# Calculate the combined loss (content loss + style loss)


# Declare Optimizer and minimize the loss


# Initialize all Variables using session and start Training

        
# You can use ---scipy.misc.imsave() for saving the image