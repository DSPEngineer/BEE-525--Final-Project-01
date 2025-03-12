#! /usr/bin/python

######################################################################
## Please use the following code to test the CNN code			    ##
## Created by Madhava Vemuri 										##
## Date 3/3/25														##
## Please use the following circuit diagram in page 5 to make the 	##
## connections. Double check the connections before you turn on Pi  ##
######################################################################
import pypic as cam

import tensorflow as tf 			# import the tensorflow library
from CNN_model.model import create_model 	
									# import the create_model function 
									# from the CNN_model folder
import os 
import time
import cv2
import numpy as np


# Path to load the weights of the model
home_dir = os.environ['PWD']
path_to_dir = f'{home_dir}'
file_name = 'current.weights.h5'

print("Welcome BEE 525 Project 1")

latest_weight_path=f"{path_to_dir}/{file_name}"			# the location where weights are saved
print( f"Weights file: {latest_weight_path}" )

# load the MNIST dataset
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data to float type 
#x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# reshape the input data to fit the Model input size
#x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Loading an instance of the model using create_model()
model = create_model()

# Loading the weights of the model 
model.load_weights(latest_weight_path)

# Evaluate the model 
#loss, accuracy = model.evaluate(x_test,y_test)
#print('Test accuracy:', accuracy)

#########################################################################

predictions = model.predict( x_test )
print ( f"Label: [{y_test[1]}]" )
print (predictions[1])
print ( f"Predict: [{np.argmax(predictions[1])}]" )

path_to_image = r"testImages/Crop_1.png"
#path_to_image = r"testImages/Crop_2.png"#path_to_image = r"testImages/Crop_5.png"
#path_to_image = r"testImages/Crop_8.png"
#path_to_image = r"testImages/Crop_9.png"
startCapture = time.time()

capturedImage = 'pic-01.jpg'
cam = cam.pypic(  file=capturedImage )
cam.capture()

# Loading the grayscale image from a path to an array ?image?
#image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(  cam.getImgFullName(), cv2.IMREAD_GRAYSCALE )
# To print the shape of the image using shape() method
print( f"The shape of image array is {image.shape}")
# The image shape is : (1944, 2592)

# Trying the resize() method
image_reshape = cv2.resize(image,(28,28))
# printing the outcome of reshape() method
print(f"The new image shape is : {image_reshape.shape}")
# The image shape is : (28, 28)

# Trying the slice technique
# image[start_row:end_row, start_col:end_col]
## image_cropped = image[50:78, 50:78]
# printing the outcome of reshape() method
## print(f"The cropped image shape is : {image_cropped.shape}")
# The cropped image shape is : (28, 28)


# creating the test_image for a compatible image float type normalized image
# each pixel value is between (0,1)
test_image = image_reshape.astype(float)/255.0
# Expanding the dimensions using expand_dims() method in tf library
test_image_tensor = tf.expand_dims(test_image,axis=0)
print( f"Tensor shape is {test_image_tensor.shape}" )

## Compute image capture Latency (time):
endCapture = time.time()
captureLatency = endCapture - startCapture
print( f"Image capture time: {captureLatency}" ) # prints elapsed time


startPredict = time.time()
# Generating the inference using the predict() method
predictions = model.predict( [test_image_tensor] )
endPredict = time.time()
predictLatency = endPredict - startPredict

print (predictions)
print ( f"Predict: [{np.argmax(predictions)}]" )

del cam

## Compute image capture time:
width=7
precision=6
print( "--------------------------------------------------------------------------------" )
print( predictions )
print( "-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --" )
print( f"              Predict: [{np.argmax(predictions)}]" )
print( f"   Image capture time: {1000*captureLatency:>{width}.{precision}f} [ms]" ) # prints elapsed time
print( f"Image prediction time: {1000*predictLatency:>{width}.{precision}f} [ms]" ) # prints elapsed time
print( "                       -----------------------" )
print( f"           Total time: {1000*(captureLatency+predictLatency):>{width}.{precision}} [ms]" ) # prints elapsed time


# end = time.time()
# print(end - start) # prints elapsed time
