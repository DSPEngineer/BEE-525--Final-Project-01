#! /usr/bin/python

import cv2
import numpy as np
from picamera2 import Picamera2, Preview
import time

##init
picam2 = Picamera2()

camera_config = picam2.create_still_configuration(main={ "size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")

picam2.configure(camera_config)

picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(2)
picam2.capture_file("test.jpg")


exit(0)


from picamera2 import Picamera2, Preview
#from picamzero import Camera
from time import sleep

#from picamera import Camera
#import time
#import picamzero as pc

import numpy as np

import os

home_dir = os.environ['HOME'] #set the location of your home directory
cam = PiCamera2()
#cam.resolution(1024,768)

cam.start_preview()
sleep(5)
#cam.take_photo(f"{home_dir}/Desktop/new_image.jpg") #save the image to your desktop
cam.take_photo(f"./new_image.jpg") #save the image to your desktop
cam.stop_preview()
