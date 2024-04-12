import cv2
import numpy as np
import os
import glob
from drawCoordinates import *

def estimate2Dpose(image, mm_to_pixel,camera_matrix,dist_coeffs,rvecs,tvecs,axis_length):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0, 100])
    upper_white = np.array([255, 40, 255])
    result = cv2.inRange(hsv, lower_white, upper_white)
    blurred = cv2.GaussianBlur(result, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # threshold_area = 500
    # large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 11000<area<20000:
            M = cv2.moments(cnt)
            # print(cv2.contourArea(cnt))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # print(cX,cY)
                CX = (abs((-64/641)*cX - cY + 734.1372855))/(((-64/641)**2+1)**(1/2))
                CX = round(CX*0.2316/10,2)
                CY = (abs(645/58*cX - cY - 18161.24))/(((645/58)**2+1)**(1/2))
                CY = round(CY*0.2316/10,2)
                # print(CY,CX)
                cv2.circle(image, (cX, cY), 7, (255, 0, 0), -1)
                cv2.putText(image, f'({CY}, {CX})', (cX, cY+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
    image_with_axes = drawCoordinates(image, camera_matrix, dist_coeffs, rvecs, tvecs,axis_length)
    return image_with_axes,CY, CX


