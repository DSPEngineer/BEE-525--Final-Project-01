#! /usr/bin/python

######################################################################
## Please use the following code to test the CNN code			    ##
## Created by Madhava Vemuri 										##
## Date 3/3/25														##
## Please use the following circuit diagram in page 5 to make the 	##
## connections. Double check the connections before you turn on Pi  ##
######################################################################

import pypic as camera
import sevenSegment as ssd
from time import sleep

import tensorflow as tf 			# import the tensorflow library
from CNN_model.model import create_model 	
									# import the create_model function 
									# from the CNN_model folder
import os 
import time
import cv2
import numpy as np

## INitialze the seven segment display (needed once at beginning)
display = ssd.sevenSegment()

# Path to load the weights of the model
home_dir = os.environ['PWD']		## store weights with code to make things easy
path_to_dir = f'{home_dir}'
file_name = 'current.weights.h5'	## symLink to the desired seights file

print("                          BEE 525 Project, Winter 2025")
print("                        University of Washington, Bothell")
print("                            Jose Pagan & Vincent Dang" )
print("                              Due by: 16 March 2025" )
print( "--------------------------------------------------------------------------------" )

## CNN Model initialization, read weitghts from file.
##  This model was trained seperately and the weights were saved to a file.
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

# Loading the weights from a pre-trained model 
model.load_weights(latest_weight_path)

# Evaluate the model 
loss, accuracy = model.evaluate(x_test,y_test)
print('Model accuracy:', accuracy)

#########################################################################

## Loop and continue capturing images
captureLoop = True

while captureLoop==True:
	## This section captures a live image, via the Pi-Camera
	capturedImage = 'pic-01.jpg'				## Arbitraary file name
	cam = camera.pypic(  file=capturedImage )		## pypic is a class for my images

	cam.capture()

	## Start timing the capture after the image is aligned
	startCapture = time.time()

	# Loading the grayscale image from a path to an array ?image?
	image = cv2.imread(  cam.getImgFullName(), cv2.IMREAD_GRAYSCALE )
	# DEBUG: To print the shape of the image using shape() method
	### print( f"The shape of image array is {image.shape}")
	# The image shape should be (1944, 2592)

	# Trying the resize() method
	image_reshape = cv2.resize(image,(28,28))
	# DEBUG: printing the outcome of reshape() method
	## print(f"The new image shape is : {image_reshape.shape}")
	# The image shape should be (28, 28)

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
	## DEBUG: Lets see the tensor dimensions
	## print( f"Tensor shape is {test_image_tensor.shape}" )

	## Done with capture and image processing.
	endCapture = time.time()

	print( "================================================================================" )

	## Compute image capture Latency (time):
	captureLatency = endCapture - startCapture
	print( f"Image capture time: {captureLatency}" ) # prints elapsed time

	print( "--------------------------------------------------------------------------------" )
	### Captured Image Precition time measurement
	startPredict = time.time()

	# Generating the inference using the predict() method
	predictions = model.predict( [test_image_tensor] )

	# Prediction done!
	endPredict = time.time()
	predictLatency = endPredict - startPredict

	## evaluate the preditcion data
	predictedValue = np.argmax(predictions)

	## Set the value on the 7-segment display
	display.setDisplay( predictedValue, ssd.DP_OFF)
	display.showDisplay()

	# Time taken to predict and display on 7 segment:
	endPredict = time.time()

	# Prediction and display done!
	predictDisplayLatency = endPredict - startPredict

	## need to free the pycam object after I am done with the image
	## This also removes the image from disk.
	del cam

	## Report image capture and prediction time:
	width=7
	precision=6
	print( "-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --" )
	print( predictions )
	print( "-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --" )
	print( f"               Prediction: [{predictedValue}]" )
	print( f"       Image capture time: {1000*captureLatency:>{width}.{precision}f} [ms]" ) 		# prints elapsed time
	print( f"    Image prediction time: {1000*predictLatency:>{width}.{precision}f} [ms]" ) 		# prints elapsed time
	print( f" Predict & 7-Segment time: {1000*predictDisplayLatency:>{width}.{precision}f} [ms]" ) 	# prints elapsed time
#	print( "                       -----------------------" )
#	print( f"           Total time: {1000*(captureLatency+predictLatency):>{width}.{precision}} [ms]" ) # prints elapsed time
	print( f"     Total execution time: {1000*(captureLatency+predictDisplayLatency):>{width}.{precision}} [ms]" ) # prints elapsed time

	## Wait for input:
	action = ''
	print( "\n\nPlease select the next action:")
	while action != 'c':
		print( "   c - continue and predict another image ")
		print( "   q - quit predictions ")
		action = input( "   Enter c or q, then press enter: ")
		if action == 'q' :
			## Quit this program
			captureLoop = False
			break
		elif action != 'c' :
			print( f"\n\n*** INVALID INPUT: [{action}]")
			print( "Please enter only one of the choices bleow:")

	## clear the display after 2 seconds, after selection is made
	sleep(2)
	display.setDisplay( -1, ssd.DP_OFF)
	display.showDisplay()
