#! /usr/bin/python

import tensorflow as tf
from tensorflow import keras

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from PIL import Image

def png_to_mnist(image_path):
    img = Image.open(image_path).convert('L')  # Open image and convert to grayscale
    cv.imshow ( img )
    cv.show()
    img = np.asarray(img) / 255  # Normalize pixel values
    img = np.resize(img, (28, 28 ,1))  # Resize to 28x28
    im2arr = np.array(img)
    return im2arr.reshape(1,28,28,1)

    return np.array( im2arr.reshape(1,28,28,1) )

    im2arr = im2arr.reshape(1,28,28,1)
    features_data = np.append(features_data, im2arr, axis=0)
    label_data = np.append(label_data, [image_label], axis=0)
    return img_array

imageId = 7

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
                   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
		           metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
                 )

	return model


########################################################################
########################################################################
# Path to save the weights of the model
weights_file = ".weights.h5"
weights_dir = "cp_weights"
weights_path = os.path.join( os.environ['PWD'], weights_dir, weights_file )
print(f"File: ${weights_path}")

## Load the MNIST data:
mnist = tf.keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = train_data / 255, test_data / 255

# Preprocess the data to float type 
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# reshape the input data to fit the Model input size
train_data = train_data.reshape(-1,28,28,1)
test_data  = test_data.reshape(-1,28,28,1)


## Loading a saved instance of the model using load_model()
print("Load model")
new_model = tf.keras.models.load_model( 'myModel.keras' )
# Show the model architecture
new_model.summary()

# Re-evaluate the model using the test data from MNIST
print("Evaluate model")
loss, acc = new_model.evaluate( test_data, test_labels, verbose=2)
#loss, acc = new_model.evaluate( train_data, train_labels, verbose=1)
print("** Restored model, accuracy: {:5.2f}%".format(100 * acc))
print( f"-- train_data shape: {train_data[0].shape}" )
print( "--------------------------------------------------------------------" )


print( os.listdir( 'myTestImages' ) )
tf_dim=28
tf_size=(tf_dim, tf_dim)

def get_image( fileName ) :
    img_array = cv.imread( fileName, cv.IMREAD_UNCHANGED )[:,:,0]     #Read the image as a grayscale
    print( f"-- Shape of raw image: {img_array.shape}" )
    resize_image = cv.resize(img_array, tf_size)
    print( f"-- Shape of resized image: {resize_image.shape}" )

    #,  interpolation=cv.INTER_AREA  )  #Resize the data to the MNIST dimensions
    resize_image = resize_image.astype(float) / 255.0
    resize_image_tensor = tf.expand_dims( resize_image, axis=0)
    print( f"-- Shape of tensor image: {resize_image_tensor.shape}" )
    return resize_image_tensor
    

imageDir = "myTestImages"
for img in os.listdir( imageDir ) :
    myImage = get_image(  f"{imageDir}/{img}" )
    print( myImage.shape)
#    print( myImage)

    predictions = new_model.predict( myImage )
    print( predictions )
