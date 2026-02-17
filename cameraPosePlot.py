import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# computes the camera center in units
def camera_center_world(rvec, tvec):
    rvec = np.array(rvec).reshape(3, 1)
    tvec = np.array(tvec).reshape(3, 1)
    R, _ = cv.Rodrigues(rvec)
    C = (-R.T @ tvec).reshape(3)
    return C

# computes the camera centers in meters
def compute_camera_centers(rvecs, tvecs, square_size_m=1.0):
    centers = []
    for rv, tv in zip(rvecs, tvecs):
        C_units = camera_center_world(rv, tv)
        centers.append(C_units * float(square_size_m))
    return np.vstack(centers)


# makes the plot using matplotlib
def plot_camera_centers(centers_m, labels, square_size_m, board_shape,
                        axis_len_squares):
    
    P = np.asarray(centers_m)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # camera centers
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker="o")
    for p, lab in zip(P, labels):
        if lab:
            ax.text(p[0], p[1], p[2], lab, fontsize=8)

    # board plane
    cols, rows = board_shape
    w = (cols - 1) * float(square_size_m)
    h = (rows - 1) * float(square_size_m)

    bx = np.array([0, w, w, 0, 0])
    by = np.array([0, 0, h, h, 0])
    bz = np.zeros_like(bx)
    ax.plot(bx, by, bz)

    # axes
    L = float(axis_len_squares) * float(square_size_m)
    ax.plot([0, L], [0, 0], [0, 0])
    ax.plot([0, 0], [0, L], [0, 0])
    ax.plot([0, 0], [0, 0], [0, L])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    mid = (mins + maxs) / 2.0
    span = np.maximum(maxs - mins, 1e-6)
    radius = 0.6 * span.max()

    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)

    ax.view_init(30, -60)
    plt.show()
