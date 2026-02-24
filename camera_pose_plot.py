import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# computes camera center in world coordinate
def camera_center_world(rvec, tvec):
    rvec = np.array(rvec).reshape(3, 1)
    tvec = np.array(tvec).reshape(3, 1)
    R, _ = cv.Rodrigues(rvec)
    C = (-R.T @ tvec).reshape(3)
    return R, C

# computes camera centers in units and converts to meters
def compute_camera_centers(rvecs, tvecs, square_size_m):
    centers = []
    for rvec, tvec in zip(rvecs, tvecs):
        C_units = camera_center_world(rvec, tvec)
        centers.append(C_units * square_size_m)
    return np.array(centers)

# computes pyramid vertices in world coordinate
def _camera_pyramid_world(rvec, tvec, scale):

    R, C = camera_center_world(rvec, tvec)
    R_transpose = R.T
    height = scale
    base = 0.5 * scale

    top_c = np.array([0.0, 0.0, 0.0])
    base_c = np.array([
        [-base, -base, height ],
        [ base, -base, height],
        [ base,  base, height],
        [-base,  base, height],
    ], dtype=float)


    pyramid_top = C + (R_transpose @ top_c.reshape(3, 1)).reshape(3)
    pyramid_base = C + (R_transpose @ base_c.T).T

    #flipping the z for better understanding
    pyramid_top[2] *= -1.0
    pyramid_base[:, 2] *= -1.0

    return pyramid_top, pyramid_base, R_transpose

# plots camera poses compared to the board in 3D
def plot_camera_poses(rvecs, tvecs, image_names,
                      square_size_m, board_size,
                      axis_len_squares=8.0,
                      pyramid_len_squares=6.0):

    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")

    cols, rows = board_size
    w = (cols - 1) * float(square_size_m)
    h = (rows - 1) * float(square_size_m)

    # plot the chessboard plane
    board_x = np.array([0, w, w, 0, 0])
    board_y = np.array([0, 0, h, h, 0])
    board_z = np.zeros(5)
    axis.plot(board_x, board_y, board_z)

    # plot the world axes on the chessboard plane
    L = float(axis_len_squares) * float(square_size_m)
    axis.plot([0, L], [0, 0], [0, 0])
    axis.plot([0, 0], [0, L], [0, 0])
    axis.plot([0, 0], [0, 0], [0, L])
    

    pyramid_m = float(pyramid_len_squares) * float(square_size_m)

    all_pts = []

    # plot camera pyramids and axes
    for rv, tv, name in zip(rvecs, tvecs, image_names):
        pyramid_top, pyramid_base, R_transpose = _camera_pyramid_world(rv, tv, pyramid_m)

        pyramid_faces = [
            [pyramid_top, pyramid_base[0], pyramid_base[1]],
            [pyramid_top, pyramid_base[1], pyramid_base[2]],
            [pyramid_top, pyramid_base[2], pyramid_base[3]],
            [pyramid_top, pyramid_base[3], pyramid_base[0]],
            [pyramid_base[0], pyramid_base[1], pyramid_base[2], pyramid_base[3]],
        ]

        # function to draw the pyramid faces with transparency, k is for black
        poly = Poly3DCollection(pyramid_faces, alpha=0.2, linewidths=1.0, edgecolor="k")
        axis.add_collection3d(poly)

        # draw pyramid
        loop = np.append(pyramid_base, [pyramid_base[0]], axis=0)
        axis.plot(loop[:, 0], loop[:, 1], loop[:, 2], linewidth=1.0)
        for i in range(4):
            axis.plot([pyramid_top[0], pyramid_base[i, 0]],
                    [pyramid_top[1], pyramid_base[i, 1]],
                    [pyramid_top[2], pyramid_base[i, 2]],
                    linewidth=1.0)

        # draw name
        if name:
            axis.text(pyramid_top[0], pyramid_top[1], pyramid_top[2], name, fontsize=5)

        # camera axis
        axis_len = 0.05

        axes_camera = np.array([
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ])

        dirs_world = (R_transpose @ axes_camera.T).T
        
        #flip z
        dirs_world[:, 2] *= -1.0

        axes_world = pyramid_top + dirs_world

        axis_colors = ["blue", "blue", "red"]
        
        #draw camera axes
        for i in range(3):
            axis.plot(
                [pyramid_top[0], axes_world[i, 0]],
                [pyramid_top[1], axes_world[i, 1]],
                [pyramid_top[2], axes_world[i, 2]],
                linewidth=1,
                color=axis_colors[i]
            )

        all_pts.append(pyramid_top)
        all_pts.append(pyramid_base)

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")

    # looks good from this position, can be rotated in the plot
    axis.view_init(30, -60)
    plt.show()
