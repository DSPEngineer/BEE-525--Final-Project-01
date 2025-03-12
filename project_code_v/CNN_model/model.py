######################################################################
## Please use the following code to download the MNIST dataset using## 
## keras library. 								 					##
## Created by Madhava Vemuri 										##
## Date 3/3/25														##
######################################################################

import tensorflow as tf # import the tensorflow library

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
	
	model.compile(optimizer='adam', \
		loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits =False),
		metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

	return model


if __name__=='__main__':
	
	# Creating a basic model 
	model = create_model()
	
	# Display the model's architecture 
	model.summary()
