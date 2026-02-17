import cv2 as cv
import numpy as np
import glob
import drawImage
import manualCornerInput
import cameraPosePlot

### Initialization

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
corners = []

# images
images = glob.glob('img/*.jpg')

image_names = []


for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = []
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    # Print the filename and whether corners were found
    print(fname, ret)

    if (ret == False): 
        ret, corners = manualCornerInput.manual_corner_input(img)
    
    objpoints.append(objp)
    print("Corners:\n", corners)
        
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    image_names.append(fname)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (9, 6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(0)


img = cv.imread('img1/30.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

square_size_m = 0.025
centers_m = cameraPosePlot.compute_camera_centers(rvecs, tvecs, square_size_m)
labels = [name.split("\\")[-1] for name in image_names]
cameraPosePlot.plot_camera_centers(centers_m, labels, square_size_m, (9,6), 8.0)

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
print("Camera matrix:\n", newcameramtx)

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (9,6),None)

if ret == True:
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    ret, rvec, tvec = cv.solvePnP(objp, corners2, newcameramtx, dist)

    out = drawImage.draw(dst, corners2, rvec, tvec, newcameramtx, dist)
    
    cv.imshow('img', out)
    k = cv.waitKey(0) & 0xFF
    cv.imwrite('output/cubetest.png', out)
 
cv.destroyAllWindows()