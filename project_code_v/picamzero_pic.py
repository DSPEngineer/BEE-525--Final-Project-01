#! /usr/bin/python3

from picamzero import Camera
from time import sleep
import os

class pycamzero_pic:

    def __init__(self):
        self.imgWidth  = 28
        self.imgHeight = 28
        self.imgColors = 'gray'
        self.imgDir    = os.environ['PWD']
        self.imgFile   = 'img_picamzero.png'
        self.cam       = Camera()

    def setDir(self, iDir=os.environ['PWD']):
        self.imgDir    = iDir

    def setImgName(self, iName='img_picamzero.png'):
        self.imgFile   = iName

    def setDimension(self, iWidth=28, iHeight=28, iColor='gray'):
        self.imgWidth  = iWidth
        self.imgHeight = iHeight
        self.imgColors = iColor

    def capture(self):
        self.cam.still_size   = ( self.imgWidth, self.imgHeight )
        self.cam.preview_size = ( self.imgWidth, self.imgHeight )
        self.cam.start_preview()
        sleep(8)
        self.cam.take_photo( f"{self.imgDir}/{self.imgFile}" )   #save the image
        self.cam.stop_preview()

if __name__ == '__main__':
    tmp = pycamzero_pic()
    tmp.setImgName('xxx')
    tmp.setDir()
    tmp.setDimension(1080, 740)
    tmp.capture()
