import cv2 as cv
import numpy as np
import glob

### Helper functions

# manual input of corners by clicking on the image
def click_event(event, x, y, flags, param):
    offset = 5 # offset to position O closer to the click point
    if event == cv.EVENT_LBUTTONDOWN:
        corners.append([x, y])
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img2, "O", (x - offset, y + offset), font, 0.5, (255, 0, 0), 2)
        cv.imshow('image', img2)

# draw the cube on the image
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
 
    # draw pillars
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255,255,0),1)

    # draw top axis
    img = cv.drawContours(img, [imgpts[4:]],-1,(255,255,0),1)

    # draw ground axis
    img = cv.drawContours(img, [imgpts[:4]],-1,(255,255,0),1)

    # draw top floor in green
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,255,0),-1)

    return img



### Initialization

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# cube axis points
axis = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0],
                   [0,0,-2],[0,2,-2],[2,2,-2],[2,0,-2] ])

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
corners = np.array((6 * 9, 2), dtype=np.float32)

# images
images = glob.glob('img2/*.jpg')


for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = []
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if (ret == False): 
        img2 = img.copy()
        cv.imshow('image', img2)
        cv.setMouseCallback('image', click_event)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # findChessboardCoarners returns np.array of type np.float32, so this is just for the manual input
        corners = np.array(corners, np.float32)
        # Make return true again because now we have corners
        ret = True
    
    # Print the filename and whether corners were found
    print(fname, ret)
    
    objpoints.append(objp)
        
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (9, 6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(0)

    

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread('img/30.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]


    
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (9,6),None)
 
if ret == True:
    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
 
    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
 
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
 
    img = draw(img,corners2,imgpts)
    cv.imshow('img',img)
    k = cv.waitKey(0) & 0xFF
    cv.imwrite('cube.png', img)
 
cv.destroyAllWindows()