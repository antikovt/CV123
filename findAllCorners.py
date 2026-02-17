import cv2 as cv
import numpy as np

def find_corners(img, x, y):

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((x * y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:y, 0:x].T.reshape(-1, 2)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (x, y), None)

    # If found, add object points, image points (after refining them)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        return None, None

    return ret, corners2
