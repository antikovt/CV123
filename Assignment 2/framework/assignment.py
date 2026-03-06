import glm
import random
import numpy as np
import os
import cv2

block_size = 1
SPACING = 3

# hard coded but easier for now
CAM_CONFIG_PATHS = [
    "..\\CV123\\Assignment 2\\cam1\\config.xml",
    "..\\CV123\\Assignment 2\\cam2\\config.xml",
    "..\\CV123\\Assignment 2\\cam3\\config.xml",
    "..\\CV123\\Assignment 2\\cam4\\config.xml",
]

### some helper functions

# loads the voxels look up table
def load_voxels(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    vox = fs.getNode("Voxels").mat()
    fs.release()
    return np.asarray(vox, dtype=np.float64).reshape(-1, 3)

# loads the config file
def load_config(path: str):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    rvec_node = fs.getNode("RotationVector")
    tvec_node = fs.getNode("TranslationVector")

    rvec = rvec_node.mat()
    tvec = tvec_node.mat()
    fs.release()

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    return rvec, tvec

# calculate camera center
def camera_center(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).reshape(3)
    return C, R

# swap y and z and flip y
def swap_yz(v):
    v = np.array([v[0], v[2], v[1]], dtype=np.float64)
    v[1] *= -1.0

    return v

# from a 3x3 rotation matrix to eulerian angles in radians
def euler_convert_angles(R):

    R = np.asarray(R, dtype=np.float64)
    y = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    x = np.arctan2(-R[1, 2], R[2, 2])
    z = np.arctan2(-R[0, 1], R[0, 0])

    return x, y, z

### Here are the functions used for drawing

# generates a custom plane
def generate_grid(width, depth):
    data, colors = [], []

    for x in range(width):
        for z in range(depth):
            FLOOR_Y = -1.0
            data.append([x * block_size - width/2, FLOOR_Y, z * block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])

    # added some axis for testing
    axis_len = 5

    for i in range(axis_len):
        data.append([i * block_size, 0, 0])
        colors.append([1.0, 0.0, 0.0])
        data.append([0, i * block_size, 0])
        colors.append([0.0, 1.0, 0.0])
        data.append([0, 0, i * block_size])
        colors.append([0.0, 0.0, 1.0])

    return data, colors

# voxel positions
def set_voxel_positions(width, height, depth):
    voxels = load_voxels("..\\CV123\\Assignment 2\\voxels.xml")

    # normalize coordinates for coloring
    xmin, xmax = voxels[:,0].min(), voxels[:,0].max()
    ymin, ymax = voxels[:,1].min(), voxels[:,1].max()
    zmin, zmax = voxels[:,2].min(), voxels[:,2].max()

    x_norm = (voxels[:,0] - xmin) / (xmax - xmin)
    y_norm = (voxels[:,1] - ymin) / (ymax - ymin)
    z_norm = (voxels[:,2] - zmin) / (zmax - zmin)

    # color pattern
    colors = np.stack([x_norm, z_norm, y_norm], axis=1).astype(np.float32)

    # convert coordinates to viewer system
    kept_view = np.array([swap_yz(v) for v in voxels], dtype=np.float64)

    kept_view *= SPACING

    return kept_view.tolist(), colors.tolist()


# camera positions
def get_cam_positions():

    cam_positions = []
    for path in CAM_CONFIG_PATHS:
        rvec, tvec = load_config(path)
        C_world, _R = camera_center(rvec, tvec)

        C_viewer = swap_yz(C_world)

        # apply same scaling as voxels
        C_viewer *= SPACING

        cam_positions.append(C_viewer.tolist())

    cam_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cam_positions, cam_colors

# camera orientations, figuring out this took a long time
def get_cam_rotation_matrices():
    cam_angles = []

    #world coordinate system to viewer
    P = np.array([[1, 0, 0],[0, 0, 1],[0, 1, 0]], dtype=np.float64)
    S = np.diag([1.0, -1.0, 1.0]).astype(np.float64)  # flip Y
    B = S @ P  # world -> viewer

    for path in CAM_CONFIG_PATHS:
        rvec, tvec = load_config(path)

        # rotation from world to camera coords, then transpose to obtain camera-to-world rotation, then from world coord to the viewer coords
        R_wc, _ = cv2.Rodrigues(rvec)
        R_cw = R_wc.T
        R_cv = B @ R_cw

        # axes in viewer coords
        right = R_cv @ np.array([1.0, 0.0, 0.0])
        down = R_cv @ np.array([0.0, 1.0, 0.0])
        forward = R_cv @ np.array([0.0, 0.0, 1.0])

        up  = -down

        # normalize
        forward = forward / (np.linalg.norm(forward))
        right = right / (np.linalg.norm(right))
        up = up / (np.linalg.norm(up))

        # recompute axes so they remain perpendicular and normalized
        right = np.cross(up, forward)
        right = right / (np.linalg.norm(right))
        up = np.cross(forward, right)

        # Final rotation matrix
        R_final = np.eye(3, dtype=np.float64)
        R_final[:, 0] = forward
        R_final[:, 1] = -up
        R_final[:, 2] = right

        ax, ay, az = euler_convert_angles(R_final)
        cam_angles.append([np.degrees(ax), np.degrees(ay), np.degrees(az)])

    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]

    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])

    return cam_rotations