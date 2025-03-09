#! /usr/bin/python

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

imageId = 7

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

#os.mkdir( weights_path )
#print( f"Directoy {weights_path} created." )

# Checkpoint callback to saves the model's weights after each epoch
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)

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
plt.pause( 3 )
plt.show(  block=False )

# Loading an instance of the model using create_model()
print("Load model")
new_model = tf.keras.models.load_model( 'myModel.keras' )

# Re-evaluate the model
print("Evaluate model")
loss, acc = new_model.evaluate( test_data, test_labels, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

predictions = new_model.predict(test_data)
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


