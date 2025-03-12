#! /usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf

#################################################################################
# Loading an instance of the model using create_model()
print("Load model")
model = tf.keras.models.load_model( 'myModel.keras' )

#model.summary()

# Re-evaluate the model
# print("Evaluate model")
# loss, acc = model.evaluate( test_data, test_labels, verbose=1)
# print( "Restored Model evaluation:" )
# print( "        loss: {:5.3f}%".format(100 * loss))
# print( "    accuracy: {:5.3f}%".format(100 * acc))


#################################################################################

file = r"testImages/Crop_9.png"
test_image = cv.imread(file)[:,:,0]
print( test_image.shape )

img = cv.resize( test_image, (28,28) )
img = img.astype(float)/255.0
img = tf.expand_dims( img, axis=0)

predictions = model.predict( img )
print(predictions)
print(f"Predicted Value: [{ np.argmax( predictions ) }]")
