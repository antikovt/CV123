import cv2 as cv
import numpy as np

from corner_search import find_corners, manual_corner_input
from cube_draw import draw
from remove_inaccurate_results import reprojection_error_filter_silent
from gaussian_foreground import gaussian_background_model, write_mask_video_mahalanobis, load_intrinsics

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

cam_nr = 0
while cam_nr < 1 or cam_nr > 4:
    if type(cam_nr) != int:
        break
    cam_nr = int(input("\nChoose the camera (1-4): "))
question = input("\nRecalibrate? (y/n): ")

if question == "y":
    objp = np.zeros((corners_y * corners_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

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
        # TODO: 100 is just for faster testing, change back to 26 for final run
        if frame_nr == 100:
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

    vid = cv.VideoCapture(f'4persons/cam{cam_nr}/extrinsics.avi')
    frame_nr = 0
    objpoints = []
    imgpoints = []

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
            dst = dst[ry:ry + rh, rx:rx + rw]

            ret2, coords = manual_corner_input(dst, corners_x, corners_y, subpix=True)
            if ret2:
                objpoints.append(objp)
                imgpoints.append(coords)

    vid.release()

    coords = np.mean(imgpoints, axis=0) # Finds the average from all the manual input results
    ret, rvec, tvec = cv.solvePnP(objp, coords, newcameramtx, np.zeros((5, 1)))

    R, _ = cv.Rodrigues(rvec)
    print("Saving camera extrinsics...")
    extr_filename = f'4persons/cam{cam_nr}/extrinsics.xml'
    extr_storage = cv.FileStorage()
    extr_storage.open(extr_filename, cv.FileStorage_WRITE)
    extr_storage.write('RotationVector', R)
    extr_storage.write('TranslationVector', tvec)
    extr_storage.release()



if question == "n": newcameramtx, dist = load_intrinsics(f'cam{cam_nr}/intrinsics.xml')

mean, std, newK, roi, map1, map2 = gaussian_background_model(
    f"4persons/cam{cam_nr}/background.avi",
    newcameramtx, dist,
    sample_step=5,
    max_samples=300,
)

# Observations:
# under 15 min_std shows too much of the shadows
# around 4 and 15 seem to work well for all but 2, left them like this for now
T = 4.0
min_std = 15.0
area = 750
aggressiveness = 1

if cam_nr == 2:
    T = 4.0
    min_std = 15.0

if cam_nr == 3:
    T = 4.0
    min_std = 15.0

if cam_nr == 4:
    T = 4.0
    min_std = 15.0

write_mask_video_mahalanobis(
    f"4persons/cam{cam_nr}/video.avi",
    f"4persons/cam{cam_nr}/mask.avi",
    mean, std,
    roi, map1, map2,
    T,
    min_std,
    area, aggressiveness
)

print("Wrote 4persons/cam{}/mask.avi".format(cam_nr))