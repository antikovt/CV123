import cv2 as cv
import numpy as np
import os
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
cv.ocl.setUseOpenCL(False)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def find_corners(img, x, y, SB=False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    if SB:
        ret, corners = cv.findChessboardCornersSB(gray, (x, y), flags=0)
    else:
        ret, corners = cv.findChessboardCorners(gray, (x, y), flags=cv.CALIB_CB_FAST_CHECK)

    # If found, add object points, image points (after refining them)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        return None, None

    return ret, corners2


def click_event(event, x, y, flags, param):
    offset = 5  # offset to position O closer to the click point
    if event == cv.EVENT_LBUTTONDOWN:
        param["four_corners"].append([x, y])
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(param["img2"], "O", (x - offset, y + offset), font, 0.5, (255, 0, 0), 2)
        cv.imshow('image', param["img2"])


# manual input of corners, takes the image and outputs the corners array
def manual_corner_input(img, x, y):
    img2 = img.copy()
    four_corners = []
    param = {"img2": img2, "four_corners": four_corners}

    cv.imshow("image", img2)
    cv.setMouseCallback("image", click_event, param)
    while len(four_corners) < 4:
        cv.waitKey(1)
    cv.destroyAllWindows()

    # Checking if the user performed correctly
    if len(four_corners) != 4:
        return None, None

    # Corners selected in a Z-pattern
    p00, p01 = four_corners[0], four_corners[1]
    p10, p11 = four_corners[2], four_corners[3]

    points_x = [(0,   0, p00[0]), (x-1,   0, p01[0]),
                (0, y-1, p10[0]), (x-1, y-1, p11[0])]

    points_y = [(0,   0, p00[1]), (x-1,   0, p01[1]),
                (0, y-1, p10[1]), (x-1, y-1, p11[1])]

    points = []

    for j in range(y):
        for i in range(x):
            cx = bilinear_interpolation(i, j, points_x)
            cy = bilinear_interpolation(i, j, points_y)
            points.append([cx, cy])

    # findChessboardCorners returns np.array of type np.float32, needed for cornerSubPix
    points = np.array(points, np.float32)

    # TODO: idk if I should do SubPix here, it kinda makes it worse when there's obstructions
    # yeahhh, probably not

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # points = cv.cornerSubPix(gray, points, (11, 11), (-1, -1), criteria)

    return True, points

def bilinear_interpolation(x, y, points):
    points = sorted(points)
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    return int((q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0))