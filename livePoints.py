import cv2
import cv2 as cv
from findAllCorners import find_corners

camera = cv.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    if ret:

        ret2, coords = find_corners(frame, 7, 13)

        if ret2:
            cv.drawChessboardCorners(frame, (7, 13), coords, ret2)

    cv.imshow('aaa', frame)

    if cv2.waitKey(1) & 0xFF == ord("0"):
        break
