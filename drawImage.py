import numpy as np
import cv2 as cv
import drawCube

# cube axis points
cube = np.float32([
    [ 0, 0, 0],[ 0, 2, 0],[ 2, 2, 0],[ 2, 0, 0],
    [ 0, 0,-2],[ 0, 2,-2],[ 2, 2,-2],[ 2, 0,-2]])

axis = np.float32([
    [4,0,0], [0,4,0], [0,0,-4]]).reshape(-1,3)

# draw the xyz axis on the image
def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img

# main function to draw the cube and the axis on the image
def draw(img, corners, rvecs, tvecs, K, dist):

    imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, K, dist)
    img = draw_axis(img, corners, imgpts)

    imgpts, _ = cv.projectPoints(cube, rvecs, tvecs, K, dist)
    img = drawCube.draw_cube(img, imgpts, rvecs, tvecs, cube, K, dist)

    return img