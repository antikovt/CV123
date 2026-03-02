import cv2 as cv
import numpy as np
import random as rng

from corner_search import find_corners

rng.seed(12345)

vid = cv.VideoCapture('cam4/checkerboard.avi')
vid.set(cv.CAP_PROP_FRAME_COUNT, 1)
ret, frame = vid.read()
vid.release()

gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

maxCorners = 200

# Parameters for Shi-Tomasi algorithm
qualityLevel = 0.01
minDistance = 1
blockSize = 5
gradientSize = 3
useHarrisDetector = True
k = 0.04

# Copy the source image
copy = np.copy(frame)

# Apply corner detection
corners = cv.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, None, \
                                 blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector,
                                 k=k)

# Draw corners detected
print('** Number of corners detected:', corners.shape[0])
radius = 1

winSize = (4, 4)
zeroZone = (-1, -1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

corners = cv.cornerSubPix(gray, corners, winSize, zeroZone, criteria)

for i in range(corners.shape[0]):
    cv.circle(copy, (int(corners[i, 0, 0]), int(corners[i, 0, 1])), radius,
              (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)), cv.FILLED)

# Show what you got
cv.imshow('res', copy)
cv.waitKey(0)