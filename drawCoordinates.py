import cv2
import numpy as np
import os
import glob

def drawCoordinates(image,camera_matrix,dist_coeffs,rvecs,tvecs,axis_length):
    # 3D points of the coordinate axes
    axis_points = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]).reshape(-1,3)

    # Project the 3D points to 2D image points
    img_points, _ = cv2.projectPoints(axis_points, rvecs, tvecs, camera_matrix, dist_coeffs)

    # Convert image points to integer
    img_points = img_points.astype(int)
    # print(img_points[1])
    # Draw coordinate axes on the image
    cv2.line(image, tuple(img_points[0].ravel()), tuple(img_points[1].ravel()), (0,0,255), 5)  # x-axis (red)
    cv2.line(image, tuple(img_points[0].ravel()), tuple(img_points[2].ravel()), (0,255,0), 5)  # y-axis (green)
    # cv2.line(image, tuple(img_points[0].ravel()), tuple(img_points[3].ravel()), (255,0,0), 5)  # z-axis (blue)
    cv2.putText(image, "O", (1684+20, 566-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(image, "X", (1043+20, 630-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
    cv2.putText(image, "Y", (1742 +20, 1211-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
    # cv2.circle(image,(1940, 1143),20, (255,0,0), 15)

    return image, img_points[0]
