import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



# Phase 1: Assemble the graph
# Step 1: read in data in to read_data from LinearRegression.csv file

n_samples = # calculate number of samples

# Step 2:
X =
Y =

# Step 3:
w = 
b = 

# Step 4:
Y_predicted = 

# Step 5:
loss = 

# Step 6:
optimizer = 
 
# Phase 2: Train the model
with tf.Session() as sess:
	# Step 7: 
    sess.run(tf.global_variables_initializer())
   
	


	# Step 8: train the model
    for i in range(50):
        total_loss = 0
        for x, y in data:
			# Run the optimizer and loss, store the loss value as l
            
            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss/n_samples))
        
    
    w, b = sess.run([w, b])
	
# plot the results
