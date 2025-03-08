#! /usr/bin/python

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

imageId = 2

########################################################################
## Function to create the CNN model
########################################################################
def getMaxIndex(list):
    maxIndex = -1;
    maxVal = 0

    for  i  in range( len(list) ) :
        if list[i] > maxVal:
            maxVal = list[i]
            maxIndex = i
            
    return maxIndex


########################################################################
## Function to create the CNN model
########################################################################
def create_model():
	model = tf.keras.models.Sequential([
		tf.keras.Input(shape=(28,28,1)),
		tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
		tf.keras.layers.MaxPooling2D((2,2)),
		tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
		tf.keras.layers.MaxPooling2D((2,2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(10,activation='softmax')
              ])

	model.compile( optimizer='adam', \
                       loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits =False),
		       metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
                     )

	return model


########################################################################
########################################################################
# Path to save the weights of the model
home_dir = os.environ['HOME']
path_to_dir = f'{home_dir}/Desktop/Labs/Final_Project/weights'
file_name = '/best_weights.weights.h5'

latest_weight_path=path_to_dir+file_name 
print(f"File: ${latest_weight_path}")

mnist = tf.keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = train_data / 255, test_data / 255

# Preprocess the data to float type 
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# reshape the input data to fit the Model input size
train_data = train_data.reshape(-1,28,28,1)
test_data  = test_data.reshape(-1,28,28,1)

plt.imshow(test_data[imageId], cmap="gray");
plt.axis('on')  # Turn off axis labels and ticks
plt.show()

# Loading an instance of the model using create_model()
model = create_model()

# train the model using fit() function in keras 
model.fit( train_data, train_labels, epochs=1 )

model.evaluate( test_data, test_labels)

predictions = model.predict(test_data)
np.set_printoptions(suppress=True)

print(f"Label: [{test_labels[imageId]}]")
print(predictions[imageId])

imageList = predictions[imageId]
print(f"ImageList: {imageList}")

maxVal = max( imageList )
print(f"Max Value: {maxVal}")

#imageList = imageList.tolist()
prediction = imageList.tolist().index( maxVal )
print(f"Prediction: [{prediction}]")


