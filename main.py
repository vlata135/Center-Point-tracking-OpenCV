import cv2
import numpy as np
import os
import glob
from calibrateCamera import calibrateCamera
from estimate2Dpose import *
CHECKERBOARD = (7, 7)
h = 2560
w = 1920
SQUARE_SIZE_MM = 25 
k = 0.2316

path = "photo/calib/*.jpg"
ret, mtx, dist, rvecs, tvecs = calibrateCamera(path,CHECKERBOARD,SQUARE_SIZE_MM)

# Y = [2.5,5,2.5,10,10,12.5,2.5,5,7.5,5,7.5,2.5,10,7.5,10,5,12.5,5,5,7.5,10,2.5,12.5]
Y = [2.5 , 5 , 2.5 , 10 , 10 , 12.5 , 2.5 , 5 , 7.5 , 5 , 7.5 , 2.5 , 10 , 7.5 , 10 , 5 , 12.5 , 2.5 , 12.5 , 5 , 5 , 7.5 , 10 , 2.5 , 12.5]
X = [10  , 5  , 12.5  , 12.5  , 5  , 5  , 10  , 2.5  , 7.5  , 7.5  , 7.5  , 12.5  , 12.5  , 12.5  , 12.5  , 12.5  , 12.5  , 12.5  , 12.5  , 12.5  , 7.5  , 7.5  , 12.5  , 12.5  , 5]
errorX = []
errorY = []
count = 0
path_test = "photo/test/test_1/*.jpg"
images_test = glob.glob(path_test)
for image_test in images_test:
    image = cv2.imread(image_test)
    image,CX,CY = estimate2Dpose(image, k, mtx, dist, rvecs[0], tvecs[0], 150)
    errorX.append(abs(CX - X[count]))
    errorY.append(abs(CY - Y[count]))
    count = count + 1

avr_error_x = 0
avr_error_y = 0
for eX in errorX:
    avr_error_x = avr_error_x + eX
avr_error_x = round(avr_error_x/25, 4)
for eY in errorY:
    avr_error_y = avr_error_y + eY
avr_error_y = round(avr_error_y/25,4)
print("Sai so theo truc X")
print(avr_error_x)
print("Sai so theo truc Y")
print(avr_error_y)