#! /usr/bin/python3

from picamzero import Camera
from time import sleep

import numpy as np
import os


home_dir = os.environ['PWD'] #set the location of your current directory
cam = Camera()
#cam.still_size = (2592, 1944)
cam.preview_size = (1920, 1080)
#cam.still_size = (1024, 768)
#cam.still_size = (960, 540)
#cam.still_size = (28, 28)


cam.start_preview()
sleep(8)
cam.take_photo( f"image_picamzero.jpg" )   #save the image to your desktop
cam.stop_preview()

