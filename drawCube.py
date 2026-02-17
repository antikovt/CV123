# draw_cube.py
import numpy as np
import cv2 as cv

# draws the lines of the cube
def draw_lines(img, pts2d):
    
    # draw pillars
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(pts2d[i]), tuple(pts2d[j]),(255,255,0),1)

    # draw top axis
    img = cv.drawContours(img, [pts2d[4:]],-1,(255,255,0),1)

    # draw ground axis
    img = cv.drawContours(img, [pts2d[:4]],-1,(255,255,0),1)

    return img


# draws the cube on the image, the top polygon and the dot with distance text
def draw_cube(img, imgpts, rvec, tvec, cube_3d, K, dist):

    pts2d = np.int32(imgpts).reshape(-1, 2)
    top2d = pts2d[4:8]

    img = draw_lines(img, pts2d)

    center3d_units = cube_3d[4:8].mean(axis=0).reshape(3, 1)

    # To remember: Rodrigues gets rotation matrix
    R, _ = cv.Rodrigues(rvec)
    center_cam = np.dot(R, center3d_units) + tvec

    # convert distance to meters (1 square side is 0.025 meters)
    dist_units = float(np.linalg.norm(center_cam))
    dist_m = dist_units * 0.025

    # intensity based on distance: 255 at 0m, 0 at 4m
    V = int(np.clip(round(255.0 * ((4.0 - dist_m) / 4.0)), 0, 255))

    n_obj = np.array([[0.0], [0.0], [-1.0]])
    n_cam = np.dot(R, n_obj)
    
    z_cam = np.array([[0.0], [0.0], [1.0]])

    # orientation between camera and board normals
    angle = np.dot(n_cam.T, z_cam) / (np.linalg.norm(n_cam) * np.linalg.norm(z_cam))
    angle = float(np.clip(np.abs(angle).item(), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(angle)))

    # OpenCV hue is between 0 and 179 so we used 179 as max not 255 like the assignment description states.
    H = float(np.clip(179.0 * ((45.0 - angle_deg) / 45.0), 0.0, 179.0))

    S = 255

    # convert HSV to BGR for drawing
    hsv = np.uint8([[[H, S, V]]])
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)[0, 0].tolist()

    cv.fillConvexPoly(img, top2d, bgr)

    center2d, _ = cv.projectPoints(center3d_units.reshape(1, 3), rvec, tvec, K, dist)
    center = np.int32(center2d).reshape(-1, 2)[0]

    offset = 8

    cv.circle(img, center, 4, (0, 0, 0), -1)
    label = f"{dist_m:.2f} m"
    cv.putText(img, label, (center[0] + offset, center[1] - offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,1)

    return img
