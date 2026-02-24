import cv2 as cv
import numpy as np

from corner_search import find_corners, manual_corner_input
from cube_draw import draw

# read checkerboard characteristics from checkerboard.xml
cb_storage = cv.FileStorage()
cb_storage.open('checkerboard.xml', cv.FileStorage_READ)
cb_nodes = ['CheckerBoardWidth', 'CheckerBoardHeight', 'CheckerBoardSquareSize']
cb_values = []

for node in cb_nodes:
    cb_values.append(int(cb_storage.getNode(node).real()))
corners_x, corners_y, cell_size = cb_values

objp = np.zeros((corners_y * corners_x, 3), np.float32)
objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

vid = cv.VideoCapture('cam4/intrinsics.avi')
frame_nr = 0

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    if frame_nr == 0:
        sample_image = frame
    frame_nr += 1

    # analyze each 25th frame only
    if frame_nr == 26:
        frame_nr = 1
        ret2, coords = find_corners(frame, corners_x, corners_y, True)
        if ret2:
            objpoints.append(objp)
            imgpoints.append(coords)

vid.release()
print(len(imgpoints))

