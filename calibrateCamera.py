import cv2
import numpy as np
import os
import glob

def calibrateCamera(path, CHECKERBOARD, SQUARE_SIZE_MM):
    # CHECKERBOARD = (7, 7)
    h = 2560
    w = 1920
    # SQUARE_SIZE_MM = 25  
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM  

    prev_img_shape = None

    images = glob.glob(path)
    for fname in images:
        img = cv2.imread(fname)
        img = cv2.GaussianBlur(img,(7,7),0)
        lwr = np.array([90, 80, 70])
        upr = np.array([120, 300, 200])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(hsv, lwr, upr)
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
        dlt = cv2.dilate(msk, krn, iterations=5)
        res = 255 - cv2.bitwise_and(dlt, msk)
        res = np.uint8(res)
        ret, corners = cv2.findChessboardCorners(res, (7, 7),
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                cv2.CALIB_CB_FAST_CHECK +
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            objpoints.append(objp)
            # Refining pixel coordinates for given 2D points.
            corners2 = cv2.cornerSubPix(res, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # print(img.shape[:2])
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    return ret, mtx, dist, rvecs, tvecs