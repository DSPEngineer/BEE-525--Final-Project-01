#! /usr/bin/python
######################################################################
## Please use the following code to capture an image using the Rpi  ##
## camera connected using the camera interface. 		    ##
## Created by Madhava Vemuri 					    ##
## Date 3/3/25							    ##
## Please use the following circuit diagram in page 5 to make the   ##
## connections. Double check the connections before you turn on Pi  ##
######################################################################
#
# Please make sure the picamzero library is intalled before you import 
# the library.
#
# to install the library use the following script in the command line 
# before running this code 
# sudo apt-get install python3-picamzero
 


#from picamera2 import Camera # import the library to handle the camera
from picamzero import Camera # import the library to handle the camera
import os 
from time import sleep 		 # import the sleep function 

# Path to the current directory
home_dir = os.environ['PWD']

# Path to desitation folder 
path_to_dir = f'{home_dir}'
file_name = 'image_capture_image.jpg'

# Create a object instance for the Camera() class 
cam = Camera()

# Create a preview of view before taking the picture 
cam.start_preview()

# Wait for 4 seconds before taking the picture. Use this time to adjust 
# the view in the preview
sleep(8)

# Use the take_photo() to Capture the photo and save it in the 
# path_to_dir location with file_name
cam.take_photo(f"{path_to_dir}/{file_name}")


# End the preview after taking the picture 
cam.stop_preview()
