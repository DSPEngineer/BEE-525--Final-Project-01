#! /usr/bin/python3

######################################################################
## This file defines a class to write the 7 segment LED display on  ##
## the Raspberry Pi 4B, using GPIO pins,as listed below.            ##
##                                                                  ##
##  Created by Jose Pagan                                           ##
##    based on file from Madhava Vemuri				    ##
## Date 03/09/2025						    ##
######################################################################
## Hithub:                                                          ##
##    https://github.com/DSPEngineer/BEE-525--Final-Project-01      ##
##                                                                  ##
######################################################################

import RPi.GPIO as GPIO 	#Import the Raspberry Pi GPIO Library


# Data for segment values, without the decimal point
displayPattern = {
   #### a b c d e f g
   ' ':(0,0,0,0,0,0,0),
     0:(1,1,1,1,1,1,0),
     1:(0,1,1,0,0,0,0),
     2:(1,1,0,1,1,0,1),
     3:(1,1,1,1,0,0,1),
     4:(0,1,1,0,0,1,1),
     5:(1,0,1,1,0,1,1),
     6:(1,0,1,1,1,1,1),
     7:(1,1,1,0,0,0,0),
     8:(1,1,1,1,1,1,1),
     9:(1,1,1,1,0,1,1)
}
 
# 7 Segment Display has 7 pins connected to the cathode of the LEDs 
# Here is the breakdown of the pins used to connect the LED
LED_pin_a  = 17 # Even though the it is pin 11, based on the GPIO pin digram it is pin 17
LED_pin_b  = 18 # LED cathode to b  is connected to GPIO 18 which is pin 12
LED_pin_c  = 27 # LED cathode to c  is connected to GPIO 27 which is pin 13
LED_pin_d  = 22 # LED cathode to d  is connected to GPIO 22 which is pin 15
LED_pin_e  = 23 # LED cathode to e  is connected to GPIO 23 which is pin 18
LED_pin_f  = 24 # LED cathode to f  is connected to GPIO 24 which is pin 18
LED_pin_g  = 25 # LED cathode to g  is connected to GPIO 25 which is pin 22
LED_pin_DP = 26 # LED cathode to DP is connected to GPIO 26 which is pin 37

## Array of pins for indexing
ledPins = [ LED_pin_a, LED_pin_b, LED_pin_c, LED_pin_d, LED_pin_e, LED_pin_f, LED_pin_g ] #, LED_pin_DP ]

# Constants for decimal point 
DP_OFF     = 0  # Decimal point OFF
DP_ON      = 1  # Decimal point ON

class sevenSegment():

    ## Initialization of display properties
    def __init__(self):
        self.val        = -1
        self.segPattern = displayPattern[' ']       # initialize all segments off
        self.DP         = DP_OFF
        # Set the PIN mode to either BCM mode otherwise set D mofethe value is  
        GPIO.setmode(GPIO.BCM)                      # set stored in to BCM mode
        # Setup all of the pins as output (mode)
        for pinIdx in range( len(ledPins) ):
            GPIO.setup(ledPins[pinIdx], GPIO.OUT)   # set each GPIO as OUTPUT
        GPIO.setup(LED_pin_DP, GPIO.OUT)            # set the Decimal Point GPIO as OUTPUT
        self.showDisplay()                          #  use the display method to illuminate the segments

    # Call this method to change the value to be displayed.
    # This method does not modify what is displayed. 
    def setDisplay(self, newVal, dpVal=DP_OFF):
        if ( newVal < 0 ) | (newVal > 9):
            # Error, value is out of range, clear all segments
            self.val        = -1
            self.segPattern = displayPattern[' ']
            self.DP         = DP_OFF
        else:
            # Value is in range
            self.val        = newVal
            self.segPattern = displayPattern[newVal]
            self.DP         = DP_OFF if dpVal==DP_OFF else DP_ON

    # set the segments and decimal point of the display. 
    def showDisplay(self, dpVal=-1):
        ledSegment = 0
        for ledSegment in range( len(ledPins) ):
            ## print(f"    LED_pin_{ chr( ord('a') + ledSegment ) } = {ledPins[ledSegment]} : {self.segPattern[ledSegment]}")
            GPIO.output(ledPins[ledSegment], self.segPattern[ledSegment])
        # separate handling of the decimal point. Use "dpVal" as flag to override the class value, using the
        # following algorithm:
        # If dpVal is provided, set DP_OFF when its value equals DP_OFF,
        # otherwise set DP_ON, if the value is other than DP_OFF.
        # If dpVal is not provided, usee the "self.DP" value stored in the class.
        showDP = self.DP if dpVal==-1 else ( DP_OFF if dpVal == DP_OFF else DP_ON )
        GPIO.output(LED_pin_DP, showDP)


## Unit Test Code:
if __name__ == '__main__':
    from time import sleep

    print(f"' ': {displayPattern[' ']}" )
    for i in range(10):
      print(f" {i}: {displayPattern[i]}")

    myDigit = sevenSegment()

    for i in range(10):
        myDigit.setDisplay(i)
        myDigit.showDisplay()
        sleep(3)
        
    myDigit.setDisplay(-1)
    myDigit.showDisplay()
    print(ledPins)
