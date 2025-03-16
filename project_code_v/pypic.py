#! /usr/bin/python3
######################################################################
## Wrapper class for pycamzero.                                     ##
##                                                                  ##
######################################################################
## Hithub:                                                          ##
##    https://github.com/DSPEngineer/BEE-525--Final-Project-01      ##
##                                                                  ##
######################################################################

from picamzero import Camera
from time import sleep
import os

class pypic:
    cam = None

    def __del__(self):
        os.remove( self.getImgFullName() )
        del self.cam

    def __init__(self, file=None, path=None):
        ## Ensure cam is gone to prevent re-allocating
        if self.cam != None :
            self.cam = None
            sleep(1)
        ## Use default image file name if not provided 
        if file != None :
            self.setImgName( file )
        else:
            self.setImgName()
        ## Use default image directory if not provided 
        if path != None :
            self.setImgDir( path )
        else:
            self.setImgDir()
        ## Continue class initialization             
        self.cam = Camera()         # alocate camera object
        self.setDimension()         # set opject dimension
 
    def getImgFullName(self):
        return f"{self.imgDir}/{self.imgFile}"

    def getImgDir(self, path=os.environ['PWD']):
        return self.imgDir

    def setImgDir(self, path=os.environ['PWD']):
        self.imgDir    = path

    def getImgName(self):
        return {self.imgFile}

    def setImgName(self, file='img_picamzero.png'):
        self.imgFile   = file

    def setDimension(self, iWidth=640, iHeight=480, iColor='gray'):
        self.imgWidth  = iWidth
        self.imgHeight = iHeight
        self.imgColors = iColor
        self.cam.preview_size = ( self.imgWidth, self.imgHeight )
        self.cam.still_size   = ( self.imgWidth, self.imgHeight )

    def capture(self):
        self.cam.start_preview()
        sleep(8)
        self.cam.take_photo( self.getImgFullName() )   #save the image
        self.cam.stop_preview()



if __name__ == '__main__':
    tmp = pypic( )
    tmp.setImgName('xxx-0.jpg')
    tmp.setImgDir()
    tmp.setDimension(1080, 740)
    tmp.capture()
    print( f"Image: {tmp.getImgFullName()}")
    
    ## Need to destroy the camera object between captures
    # #tmp = None
    del tmp

    tmp = pypic( file='xxx-1.jpg' )
    print( f"Image: {tmp.getImgFullName()}")
    tmp.setDimension()
    tmp.capture()
