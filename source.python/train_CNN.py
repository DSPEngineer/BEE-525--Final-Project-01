#######################################################################
## Please use the following code to download the MNIST dataset using ## 
## keras library. 						     ##
##                                                                   ##
## Created by Madhava Vemuri 					     ##
## Date 3/3/25	     						     ##
## Please use the following circuit diagram in page 5 to make the    ##
## connections. Double check the connections before you turn on Pi   ##
#######################################################################

import tensorflow as tf # import the tensorflow library
from CNN_model.model import create_model 	
									# import the create_model function 
									# from the CNN_model folder
import os 

# Path to save the weights of the model
home_dir = os.environ['HOME']
path_to_dir = f'{home_dir}/Desktop/Labs/Final_Project/weights'
file_name = '/best_weights.weights.h5'

latest_weight_path=path_to_dir+file_name 
								# is the location best weights are saved

# load the MNIST dataset
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data to float type 
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# reshape the input data to fit the Model input size
x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Loading an instance of the model using create_model()
model = create_model()

# train the model using fit() function in keras 
model.fit(x_train,y_train,epochs=1)

# Save the model weights after training using save_weights
model.save_weights(latest_weight_path)

# Evaluate the model 
loss, accuracy = model.evaluate(x_test,y_test)
print('Test accuracy:', accuracy)


