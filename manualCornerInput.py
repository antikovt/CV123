import cv2 as cv
import numpy as np

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

def bilinear_interpolation(x, y, points):
    points = sorted(points)
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

# manual input of corners by clicking on the image
def click_event(event, x, y, flags, param):
    offset = 5 # offset to position O closer to the click point
    if event == cv.EVENT_LBUTTONDOWN:
        param["corners"].append([x, y])
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(param["img2"], "O", (x - offset, y + offset), font, 0.5, (255, 0, 0), 2)
        cv.imshow('image', param["img2"])

# manual input of corners, takes the image and outputs the corners array
def manual_corner_input(img):

    img2 = img.copy()
    four_corners = []
    param = {"img2": img2, "corners": four_corners}


    cv.imshow("image", img2)
    cv.setMouseCallback("image", click_event, param)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    corners = []

    top_left = four_corners[0]
    top_right = four_corners[1]
    bottom_left = four_corners[2]
    bottom_right = four_corners[3]

    for j in range(6):
        for i in range(9):
            x = int(bilinear_interpolation(i, j, [(0, 0, top_left[0]), (8, 0, top_right[0]), (0, 5, bottom_left[0]), (8, 5, bottom_right[0])]))
            y = int(bilinear_interpolation(i, j, [(0, 0, top_left[1]), (8, 0, top_right[1]), (0, 5, bottom_left[1]), (8, 5, bottom_right[1])]))
            corners.append([x, y])

    print ("Corners:\n", corners)

    # findChessboardCorners returns np.array of type np.float32, needed for cornerSubPix
    corners = np.array(corners, np.float32)
    # Make return true again because now we have corners
    return True, np.array(corners, np.float32)

