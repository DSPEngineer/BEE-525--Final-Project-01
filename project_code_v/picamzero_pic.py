#! /usr/bin/python3

from picamzero import Camera
from time import sleep
import os

class pycamzero_pic:
    cam = None

    def __del__(self):
        del self.cam

    def __init__(self):
        if self.cam != None :
            self.cam = None
            sleep(1)
        self.cam = Camera()
        self.setDimension()
        self.setDir()
        self.setImgName()
 
    def setDir(self, iDir=os.environ['PWD']):
        self.imgDir    = iDir

    def setImgName(self, iName='img_picamzero.png'):
        self.imgFile   = iName

    def setDimension(self, iWidth=640, iHeight=480, iColor='gray'):
        self.imgWidth  = iWidth
        self.imgHeight = iHeight
        self.imgColors = iColor
        self.cam.preview_size = ( self.imgWidth, self.imgHeight )
        self.cam.still_size   = ( self.imgWidth, self.imgHeight )

    def capture(self):
        self.cam.start_preview()
        sleep(8)
        self.cam.take_photo( f"{self.imgDir}/{self.imgFile}" )   #save the image
        self.cam.stop_preview()

if __name__ == '__main__':
    tmp = pycamzero_pic()
    tmp.setImgName('xxx-0.png')
    tmp.setDir()
    tmp.setDimension(1080, 740)
    tmp.capture()
    
    ## Need to destroy the camera object between captures
    # #tmp = None
    del tmp

    tmp = pycamzero_pic()
    tmp.setImgName('xxx-1.png')
    tmp.setDimension()
    tmp.capture()
