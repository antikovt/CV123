import cv2 as cv
import numpy as np

# voxel grid size and step count
XMIN, XMAX = -4.0,  8.0
YMIN, YMAX = -7.5,  5.0
ZMIN, ZMAX = -12.2, 0.0
VOXEL_STEP = 0.3

# frame at sec 3 seems to look the best
T_SEC = 3.0

# played around with this because it seems that storing the video makes some of the white pixels darker
THRESHOLD = 254

# loads the camera configuration from the .xml file
def load_config(path):
    fs = cv.FileStorage(path, cv.FileStorage_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open config: {path}")

    K = fs.getNode("CameraMatrix").mat()
    dist = fs.getNode("CameraDistortion").mat()
    rvec = fs.getNode("RotationVector").mat()
    tvec = fs.getNode("TranslationVector").mat()

    roi_node = fs.getNode("ROI")

    rx, ry, rw, rh = roi_node.mat().flatten().astype(int).tolist()
    fs.release()

    return K, dist, rvec, tvec, rx, ry, rw, rh

# reads frame 150 from each of the camera masks( second 3)
def read_mask_frame(path, frame_index = 150, threshold = 1):

    cap = cv.VideoCapture(path)

    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_index = int(np.clip(frame_index, 0, n - 1))

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    _, frame = cap.read()
    cap.release()

    # make mask grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = (gray >= threshold).astype(np.uint8)

    return mask, frame_index

# makes the custom grid to be filled with voxels on and off voxels
def create_voxel_grid():
    xs = np.arange(XMIN, XMAX + 1e-12, VOXEL_STEP, dtype=np.float64)
    ys = np.arange(YMIN, YMAX + 1e-12, VOXEL_STEP, dtype=np.float64)
    zs = np.arange(ZMIN, ZMAX + 1e-12, VOXEL_STEP, dtype=np.float64)

    x,y,z = np.meshgrid(xs, ys, zs, indexing="xy")
    voxels = np.stack([x,y,z], axis=-1).reshape(-1, 3)

    return voxels

# saves only the on voxels
def save_carved_voxels_xml(path, voxels_world, world):
    fs = cv.FileStorage(path, cv.FileStorage_WRITE)
    fs.write("Voxels", voxels_world.astype(np.float32))

    for k, v in world.items():
        fs.write(k, float(v))

    fs.release()
    print(f"Saved voxels: {path} (total size={voxels_world.shape[0]})")

# this is the main to run this file, outputs voxels.xml
if __name__ == "__main__":

    voxels = create_voxel_grid()
    size = voxels.shape[0]
    obj_pts = voxels.reshape(-1, 1, 3).astype(np.float64)

    # count how many cameras each voxel is inside / foreground
    inside_counter = np.zeros(size, dtype=np.int16)
    voxel_camera_counter = np.zeros(size, dtype=np.int16)

    for cam in range(1, 5):
        print(f"\nCam{cam}")

        K, dist, rvec, tvec, rx, ry, rw, rh = load_config(f"cam{cam}/config.xml")

        # need to crop it again to be similar to the masks
        K_roi = K.copy()
        K_roi[0, 2] -= rx
        K_roi[1, 2] -= ry

        mask, fi = read_mask_frame(f"cam{cam}/mask.avi", frame_index=150, threshold=THRESHOLD)
        height, width = mask.shape[:2]

        print("roi (x,y,w,h):", rx, ry, rw, rh, " mask (h,w):", mask.shape[0], mask.shape[1])

        img_pts, _ = cv.projectPoints(obj_pts, rvec, tvec, K_roi, dist)
        img_pts = img_pts.reshape(-1, 2)

        x_pixel = np.round(img_pts[:, 0]).astype(np.int32)
        y_pixel = np.round(img_pts[:, 1]).astype(np.int32)

        inside = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
        inside_counter[inside] += 1

        foreground = np.zeros(size, dtype=bool)
        foreground[inside] = (mask[y_pixel[inside], x_pixel[inside]] > 0)
        voxel_camera_counter[foreground] += 1

    # if the voxel is present in all cameras, keep it
    inside_all = (inside_counter == 4)
    foreground_all = (voxel_camera_counter == 4)
    keep = inside_all & foreground_all

    kept = voxels[keep]
    print(f"\nKept voxels: {kept.shape[0]} / {size}")

    # this is to check what grid bounds would be better, added 2 to each max, substracted 2 for min
    if kept.shape[0] > 0:
        xmin, ymin, zmin = kept.min(axis=0)
        xmax, ymax, zmax = kept.max(axis=0)

        print(f"X: min = {xmin:.2f} max = {xmax:.2f}")
        print(f"Y: min = {ymin:.2f} max = {ymax:.2f}")
        print(f"Z: min = {zmin:.2f} max = {zmax:.2f}")
    
    # saving
    save_carved_voxels_xml(
        "voxels.xml",
        kept,
        world=dict(
            XMIN=XMIN, XMAX=XMAX,
            YMIN=YMIN, YMAX=YMAX,
            ZMIN=ZMIN, ZMAX=ZMAX,
            VOXEL_STEP=VOXEL_STEP,
            T_SEC=T_SEC
        )
    )