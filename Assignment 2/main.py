import cv2 as cv
import numpy as np

from corner_search import find_corners, manual_corner_input
from cube_draw import draw
from remove_inaccurate_results import reprojection_error_filter_silent

# disabled OpenCL to avoid errors/ now everything is done by the cpu
cv.ocl.setUseOpenCL(False)
cv.setUseOptimized(True)  

# read checkerboard characteristics from checkerboard.xml
cb_storage = cv.FileStorage()
cb_storage.open('checkerboard.xml', cv.FileStorage_READ)
cb_nodes = ['CheckerBoardWidth', 'CheckerBoardHeight', 'CheckerBoardSquareSize']
cb_values = []

for node in cb_nodes:
    cb_values.append(int(cb_storage.getNode(node).real()))
corners_x, corners_y, cell_size = cb_values
cell_size *= 0.001

cb_storage.release()

objp = np.zeros((corners_y * corners_x, 3), np.float32)
objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cam_nr = 0
while cam_nr < 1 or cam_nr > 4:
    cam_nr = int(input("\nChoose the camera (1-4): "))

vid = cv.VideoCapture(f'cam{cam_nr}/intrinsics.avi')
frame_nr = 0

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    if frame_nr == 0:
        sample_image = frame
    frame_nr += 1

    # analyze each 25th frame only
    # TODO: 100 is just for faster testing, change back to 25 for final run
    if frame_nr == 25:
        frame_nr = 1
        ret2, coords = find_corners(frame, corners_x, corners_y, True)
        if ret2:
            objpoints.append(objp)
            imgpoints.append(coords)

vid.release()
print(len(imgpoints))

sample_gray = cv.cvtColor(sample_image, cv.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    list(filter(lambda x: x is not None, objpoints)),
    list(filter(lambda x: x is not None, imgpoints)),
    sample_gray.shape[::-1], None, None)

h, w = sample_gray.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# helper code from last assignment, also used to filter out the selected frames with a big reprojection error
print("\nK")
print(mtx)

print("\nK optimal:")
print(newcameramtx)

while True:
    user_input = input("\nWould you like to filter out inaccurate results based on reprojection error? (y/n): ")


    if user_input == "y":
        threshold_px = float(input("\nEnter reprojection error threshold in pixels (e.g. 0.1): "))

        obj_keep, img_keep = reprojection_error_filter_silent(
            objpoints, imgpoints,
            mtx, dist, rvecs, tvecs,
            threshold_px
        )

        print(f"\nKept {len(obj_keep)} images after filtering. {len(objpoints) - len(obj_keep)} images discarded.")

        objpoints = obj_keep
        imgpoints = img_keep

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            list(filter(lambda x: x is not None, objpoints)),
            list(filter(lambda x: x is not None, imgpoints)),
            sample_gray.shape[::-1], None, None)

        h, w = sample_gray.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        print("\nK")
        print(mtx)

        print("\nK optimal:")
        print(newcameramtx)
        break
    else:
        print("\nSkipping reprojection error image filter.")
        break

print("Saving camera intrinsics...")
intr_filename = f'cam{cam_nr}/intrinsics.xml'
intr_storage = cv.FileStorage()
intr_storage.open(intr_filename, cv.FileStorage_READ)
intr_matrix = intr_storage.getNode("CameraMatrix").mat()
intr_dist = intr_storage.getNode("DistortionCoeffs").mat()
intr_storage.release()

intr_matrix = mtx
intr_dist = dist.T

# saving the intrinsics of the camera before undistortion
intr_storage.open(intr_filename, cv.FileStorage_WRITE)
intr_storage.write('CameraMatrix', intr_matrix)
intr_storage.write('DistortionCoeffs', intr_dist)
intr_storage.release()

# saving a config file to be used for the rest of the hw, the intrinsics here are for the undistorted camera
print("Saving config file...")
extr_filename = f'cam{cam_nr}/config.xml'
extr_storage_config = cv.FileStorage()
extr_storage_config.open(extr_filename, cv.FileStorage_WRITE)
extr_storage_config.write('CameraMatrix', newcameramtx)
extr_storage_config.write('CameraDistortion', np.zeros((5, 1)))

vid = cv.VideoCapture(f'cam{cam_nr}/checkerboard.avi')
frame_nr = 0
objpoints = []
imgpoints = []

# calculating extrinsics

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    frame_nr += 1

    if frame_nr == 40:
        frame_nr = 1

        # Undistorts the image
        dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
        rx, ry, rw, rh = roi
        K_roi = newcameramtx.copy()
        K_roi[0,2] -= rx
        K_roi[1,2] -= ry
        dst = dst[ry:ry + rh, rx:rx + rw]

        ret2, coords = manual_corner_input(dst, corners_x, corners_y, subpix=True)
        if ret2:
            objpoints.append(objp)
            imgpoints.append(coords)

vid.release()

coords = np.mean(imgpoints, axis=0) # Finds the average from all the manual input results

ret, rvec, tvec = cv.solvePnP(objp, coords, K_roi, np.zeros((5, 1)))

# making an extrinsics file, is not used later on
R, _ = cv.Rodrigues(rvec)
print("Saving camera extrinsics...")
extr_filename = f'cam{cam_nr}/extrinsics.xml'
extr_storage = cv.FileStorage()
extr_storage.open(extr_filename, cv.FileStorage_WRITE)
extr_storage.write('RotationMatrix', R)
extr_storage.write('TranslationVector', tvec)
extr_storage.release()

# adding extrinsics and roi to the config file, roi is really necessary because we are working with undistorted images
extr_storage_config.write('RotationVector', rvec)
extr_storage_config.write('TranslationVector', tvec)
extr_storage_config.write('ROI', roi)  
extr_storage_config.release()

# view axes to chace if calibration is fine
vid = cv.VideoCapture(f'cam{cam_nr}/checkerboard.avi')

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
    rx, ry, rw, rh = roi
    dst = dst[ry:ry + rh, rx:rx + rw]
    K_roi = newcameramtx.copy()
    K_roi[0,2] -= rx
    K_roi[1,2] -= ry

    out = draw(dst, coords, rvec, tvec, K_roi, np.zeros((5,1)), cell_size, no_cube=True)
    cv.imshow('res', out)
    cv.waitKey(1) 

