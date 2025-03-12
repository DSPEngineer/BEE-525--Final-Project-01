#! /usr/bin/python

import tensorflow as tf

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
#print( test_data[0] )

######### Train a fresh model
noEpoch=1
model = create_model()

# train the model using fit() function in keras 
#model.fit( train_data, train_labels, epochs=noEpoch, callbacks=[cp_callback] )
model.fit( train_data, train_labels, epochs=noEpoch )

### model.save('myModel.keras')
########## End of training


## Loading a saved instance of the model using load_model()
print("Load model")
new_model = tf.keras.models.load_model( 'myModel.keras' )
# Show the model architecture
new_model.summary()
# Loading an instance of the model using create_model()

# Re-evaluate the model using the test data from MNIST
print("Evaluate model")
loss, acc = new_model.evaluate( test_data, test_labels, verbose=1)
#loss, acc = new_model.evaluate( train_data, train_labels, verbose=1)
print("** Restored model, accuracy: {:5.2f}%".format(100 * acc))
print( f"-- train_data shape: {train_data[0].shape}" )
print( "--------------------------------------------------------------------" )

#################################
# Both models are ready: Compare with test data:
testImgIdx = 2
print( f"Label: {test_labels[testImgIdx]}" )
predict = new_model.predict( test_data )
print( f"    New Model: {np.argmax(predict[testImgIdx])} --- {predict[testImgIdx].shape} : {predict[testImgIdx]}" )
predict = model.predict( test_data )
print( f"    Old Model: {np.argmax(predict[testImgIdx])} --- {predict[testImgIdx].shape} : {predict[testImgIdx]}" )
#################################

print( os.listdir( 'testImages' ) )
tf_dim=28
tf_size=(tf_dim, tf_dim)

def get_image( fileName ) :
    img = Image.open( fileName ).convert('L')
    img_array = cv.imread( fileName )
    new_array = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
    new_array = cv.resize(new_array, tf_size, cv.INTER_CUBIC )
#    print(new_array.shape)
#    print(new_array)
    plt.imshow(new_array, cmap='gray')
    #plt.waitforbuttonpress()
    #plt.close('all')
    plt.show()
    new_array = new_array.astype(float) / 255.0
    resize_image_tensor = new_array.reshape(-1,28,28,1)
#    print( resize_image_tensor )
    #resize_image_tensor = tf.expand_dims( new_array, axis=0)
    return resize_image_tensor



imageDir = "testImages"
for img in os.listdir( imageDir ) :
    getImage = f"{imageDir}/{img}"
    print( f"Predicting image: {getImage}" )

    ### Get-Image()
    original_image = cv.imread( getImage )[:,:,0]
    print( original_image.shape )
 #   plt.imshow( original_image, cmap='gray' )
 #   plt.show()

    ## Resize image
    resized_image = cv.resize( original_image, (28,28) )
    print ("Resized Image:" )
    print( resized_image.shape )
#    plt.imshow( resized_image, cmap='gray' )
#    plt.show()

    img = np.array( resized_image )
    #img = np.invert( np.array( reshaped_image ) )
    print( img )
    print( img.shape )
#    plt.imshow( img, cmap='gray' )
#    plt.show()

    img = img.astype(float)/255.0
    # Expanding the dimensions using expand_dims() method in tf library
    test_image_tensor = tf.expand_dims( img, axis=0)
    print( test_image_tensor )
    print( test_image_tensor )
    ##### Done: Get_Image() 

    predict = new_model.predict( test_image_tensor )
    print( f" Loaded Model: {np.argmax(predict)} --- {predict[0].shape} : {predict[0]}" )
    predict = model.predict( test_image_tensor )
    print( f"Trained Model: {np.argmax(predict)} ---{predict[0].shape} : {predict[0]}" )
