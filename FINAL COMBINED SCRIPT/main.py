#!/usr/bin/env python3
import cv2
import cv2 as cv
import numpy as np
import glob

from corner_search import find_corners, manual_corner_input
from cube_draw import draw

def main():
    user_input = input("Which run? (1/2/3): ")
    if user_input == "1": path = 'img/run1/*.jpg'
    elif user_input == "2": path = 'img/run2/*.jpg'
    elif user_input == "3": path = 'img/run3/*.jpg'
    else:
        print("Invalid input")
        return

    print("\nAnalysing the img folder...")
    images = glob.glob(path)

    if images:
        print(f"Found {len(images)} images")
        images = sorted(images)
    else:
        print("No images found")
        return

    #-------------------------------------------------

    points_x, points_y = 9, 6
    failed_images = []
    manual_images = []

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((points_y * points_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:points_x, 0:points_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    print("Running calibration script...\n")
    for imgname in images:
        img = cv.imread(imgname)
        ret, points = find_corners(img, points_x, points_y)

        if ret:
            objpoints.append(objp)
            imgpoints.append(points)
        else:
            print(f"Chessboard corner search failed for {imgname}\n")
            user_input = input("Run manual corner selection? (y/n): ")
            if user_input != "y":
                failed_images.append(imgname)
                objpoints.append(None)
                imgpoints.append(None)
                continue
            else:
                ret, corners = manual_corner_input(img, points_x, points_y)
                if ret is None:
                    print("\nManual corner selection failed.\n")
                    failed_images.append(imgname)
                    objpoints.append(None)
                    imgpoints.append(None)
                    continue
                else:
                    print("\nManual corner selection done\n")
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    manual_images.append(imgname)

    # finds camera intrinsics for chosen set of images
    sample_image = cv.imread(images[3])
    sample_gray = cv.cvtColor(sample_image, cv.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        list(filter(lambda x: x is not None, objpoints)),
        list(filter(lambda x: x is not None, imgpoints)),
        sample_gray.shape[::-1], None, None)

    h, w = sample_gray.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    print(f"Camera calibration completed! {len(images) - len(failed_images)} images analysed.")

    # -------------------------------------------------

    print("\nChoose your next action:\n")
    print("1 - Identify corners on a live camera feed")
    print("2 - Draw cube on an image")
    print("3 - Show cube on a live camera feed")

    user_input = input("\nEnter your choice: ")
    if user_input == "1":
        live_x = int(input("\nEnter the horizontal board size: "))
        live_y = int(input("\nEnter the vertical board size: "))
        find_corners_live(live_x, live_y)

    elif user_input == "2":
        print("\nWhich image to draw cube on?")
        choice = int(input(f"Choose a number between 1 and {len(images)}: "))
        if choice < 1 or choice > len(images):
            print("\nNumber out of range.")
            return

        # Undistorts the image
        dst = cv.undistort(cv.imread(images[choice-1]), mtx, dist, None, newcameramtx)
        rx, ry, rw, rh = roi
        dst = dst[ry:ry + rh, rx:rx + rw]

        if images[choice-1] in manual_images or images[choice-1] in failed_images:
            print("\nBefore drawing a cube, image points need to be identified again after image calibration.")
            print("This image has failed to process automatically, so corners need to be manually identified.")
            second_input = input("\nRun another manual corner selection? (y/n): ")
            if second_input != "y":
                return
            else:
                ret_cube, corners_cube = manual_corner_input(dst, points_x, points_y)
        else:
            ret_cube, corners_cube = find_corners(dst, points_x, points_y)

        if ret_cube:    # Finds extrinsics, draws a cube on an undistorted image
            ret, rvec, tvec = cv.solvePnP(objp, corners_cube, newcameramtx, dist)
            out = draw(dst, corners_cube, rvec, tvec, newcameramtx, dist, 0.025)
            cv.imshow(f'Image {images[choice-1]} with a cube', out)
            cv.waitKey(0)
        else:
            print("\nHuh, something went wrong.")

    elif user_input == "3":
        live_x = int(input("\nEnter the horizontal board size: "))
        live_y = int(input("\nEnter the vertical board size: "))
        size = float(input("\nEnter the size of a single board cell (in millimeters, decimals are allowed): ")) * 0.001
        live_cube(live_x, live_y, size)


def find_corners_live(x, y):
    camera = cv.VideoCapture(0)

    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            ret2, coords = find_corners(frame, x, y)
            if ret2:
                cv.drawChessboardCorners(frame, (x, y), coords, ret2)

        cv.imshow('Press "0" to close', frame)

        if cv.waitKey(1) & 0xFF == ord("0"):
            break

def live_cube(x, y, size):
    camera = cv.VideoCapture(0)
    objp = np.zeros((y * x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    calibrated = False

    while camera.isOpened():
        ret, frame = camera.read()
        if ret and not calibrated:  # Calibrate the camera based on the first 25 frames with detectable chessboard corners, then stop
            ret2, corners = find_corners(frame, x, y)
            if ret2 and len(imgpoints) < 25:
                objpoints.append(objp)
                imgpoints.append(corners)
            elif len(imgpoints) == 25 and not calibrated:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,
                                                                  gray.shape[::-1], None, None)
                h, w = gray.shape[:2]
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                calibrated = True

        if ret and calibrated:  # Undistort the image, find adjusted corners, draw cube
            dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
            rx, ry, rw, rh = roi
            cut = dst[ry:ry + rh, rx:rx + rw]
            ret_cube, corners_cube = find_corners(cut, x, y)
            if ret_cube:
                ret, rvec, tvec = cv.solvePnP(objp, corners_cube, newcameramtx, dist)
                cubed = draw(cut, corners_cube, rvec, tvec, newcameramtx, dist, size)
            else:   # Will skip drawing the cube if corner search failed
                cubed = cut


        if not calibrated:
            cv.imshow('Press "0" to close', frame)
        else:
            cv.imshow('Press "0" to close', cubed)

        if cv.waitKey(1) & 0xFF == ord("0"):
            break


if __name__ == "__main__":
    main()
