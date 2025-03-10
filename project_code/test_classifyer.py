#! /usr/bin/python

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

imageId = 8  ## Looks like 6
imageId = 0  ## Looks like 7


########################################################################
## Function to create the Convolutional Neural Network model
########################################################################
def create_model_CNN():
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
                   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
		           metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
                     )

	return model


########################################################################
## Function to create the Neural Network model
########################################################################
def create_model_NN():
    model = tf.keras.models.Sequential()
    model.add( tf.keras.layers.Flatten( input_shape=(28,28) ) )
    model.add( tf.keras.layers.Dense( 128, activation='relu' ) )
    model.add( tf.keras.layers.Dense( 128, activation='relu' ) )
    model.add( tf.keras.layers.Dense( 10, activation='softmax' ) )

    model.compile( optimizer='adam',
                   loss = "sparse_categorical_crossentropy",
                   metrics = ['accuracy']
                 )

    return model



########################################################################
########################################################################
##  Read the MNIST data
mnist = tf.keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
#train_data, test_data = train_data / 255, test_data / 255

# Preprocess the data to float type 
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# reshape the input data to fit the Model input size
train_data = train_data.reshape(-1,28,28,1)
test_data  = test_data.reshape(-1,28,28,1)

#plt.imshow( test_data[imageId] )
#plt.axis('on')  # Turn off axis labels and ticks
#plt.pause(5)
#plt.show()

##### DONE with MNIST Data

#################################################################################
# Loading an instance of the model using create_model()
print("Load model")
new_model = tf.keras.models.load_model( 'myModel.keras' )

# Re-evaluate the model
print("Evaluate model")
loss, acc = new_model.evaluate( test_data, test_labels, verbose=1)
print( "Restored Model evaluation:" )
print( "        loss: {:5.3f}%".format(100 * loss))
print( "    accuracy: {:5.3f}%".format(100 * acc))


#################################################################################
file = r"myTestImages/Image_9.png"
test_image = cv.imread(file)[:,:,0]
print( test_image.shape )

plt.imshow( test_image )
plt.axis('on')  # Turn off axis labels and ticks
plt.show()

img_resized = cv.resize( test_image, (28,28) )
#print( img_resized.shape )
#print( img_resized )

plt.imshow( img_resized )
plt.axis('on')  # Turn off axis labels and ticks
plt.show()

img_resized = img_resized / 255.0
print( img_resized.shape )
print( img_resized )

plt.imshow( img_resized )
plt.axis('on')  # Turn off axis labels and ticks
plt.show()

exit(0)

img_reshaped = img_resized.reshape(28,28,1)
predictions = new_model.predict( img_reshaped )
print(predictions)

##np.set_printoptions(suppress=True)
print(predictions)

#print(f"Label: [{imageId}]")


maxVal = max( predictions )
print(f"Max Value: {maxVal}")

#imageList = imageList.tolist()
prediction = predictions.index( maxVal )
print(f"Prediction: [{prediction}]")


