import cv2 as cv
import numpy as np
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('img/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6, 9), None)

    # If found, add object points, image points (after refining them)
    print(fname, ret)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (6, 9), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(2)

    else:
        invGamma = 1.0 / 0.03
        table = np.array([((i / 255.0) ** invGamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        modify = cv.LUT(img, table)
        # modify = cv.convertScaleAbs(cv.LUT(img, table), alpha=0.5, beta=0)
        # modify = cv.LUT(cv.convertScaleAbs(img, alpha=0.3, beta=120), table)

        # lwr = np.array([0, 0, 143])
        # upr = np.array([179, 61, 252])
        # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # msk = cv.inRange(hsv, lwr, upr)
        # krn = cv.getStructuringElement(cv.MORPH_RECT, (50, 30))
        # dlt = cv.dilate(msk, krn, iterations=5)
        # res = 255 - cv.bitwise_and(dlt, msk)
        # res = np.uint8(res)

        cv.imshow("modified", modify)
        cv.waitKey()

        ret, corners = cv.findChessboardCorners(modify, (6, 9), None)

        # If found, add object points, image points (after refining them)
        print(fname, ret)
        if ret:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)


cv.destroyAllWindows()

img = cv.imread('img/1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('calibresult.png', dst)