#! /usr/bin/python

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


########################################################################
## constants and definitions
noEpoch = 1

## Path to save the weights of the model
weights_file = f"{noEpoch}.weights.h5"
weights_dir = "cp_weights"
weights_path = os.path.join( weights_dir, weights_file )
print(f"File: ${weights_path}")

## Create location for checkpoint files
if False == os.path.exists( weights_dir ) :
     os.mkdir( weights_dir )
     print( f"Directory \"./{weights_dir}\" created." )


########################################################################
## Function to find max index from list (obsolete)
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
                   loss = 'sparse_categorical_crossentropy',
                   metrics = ['accuracy']
                 )

    return model

########################################################################
########################################################################




########################################################################
## Loading MNIST data
mnist = tf.keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

print( f" Train Data: {train_data.size}" )
print( f"Train Label: {train_labels.size}" )

train_data = tf.keras.utils.normalize( train_data, axis=1)
test_data  = tf.keras.utils.normalize( test_data, axis=1)

print( f" Train Data: {train_data.size}" )
print( f"Train Label: {train_labels.size}" )

'''
#train_data, test_data = train_data / 255.0, test_data / 255.0

# Preprocess the data to float type 
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# reshape the input data to fit the Model input size
train_data = train_data.reshape(-1,28,28,1)
test_data  = test_data.reshape(-1,28,28,1)
'''

########################################################################
## Loading an instance of the model using create_model()
modelName = f"myModel-NN-E{noEpoch}.keras"

#Create the model
model = create_model_NN()

########################################################################
# Checkpoint callback to saves the model's weights after each epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path,
                                                 save_weights_only=True,
                                                 verbose=2)

# train the model using fit() function in keras
##model.fit( train_data, train_labels, epochs=noEpoch, callbacks=[cp_callback] )
model.fit( train_data, train_labels, epochs=noEpoch )

# Save the model
model.save( modelName )

# Evaluate the model using the  test data
loss, acc = model.evaluate( test_data, test_labels, verbose=1)
print( "Model evaluation:" )
print( "        loss: {:5.3f}%".format(100 * loss))
print( "    accuracy: {:5.3f}%".format(100 * acc))

exit(0)

########################################################################
## Use the model to make predictions
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


