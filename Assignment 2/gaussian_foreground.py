import cv2 as cv
import numpy as np

def load_intrinsics(path):
    fs = cv.FileStorage(path, cv.FileStorage_READ)
    K = fs.getNode("CameraMatrix").mat()
    dist = fs.getNode("DistortionCoeffs").mat()
    fs.release()
    return K, dist


def gaussian_background_model(background_avi_path, K, dist, sample_step=5, max_samples=None):

    cap = cv.VideoCapture(background_avi_path)
    ret, first = cap.read()


    h, w = first.shape[:2]
    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    map1, map2 = cv.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv.CV_16SC2)

    x, y, w_roi, h_roi = roi

    sum = np.zeros((h_roi, w_roi, 3), dtype=np.float64)
    sumsquares = np.zeros((h_roi, w_roi, 3), dtype=np.float64)
    n = 0
    frame_nr = 0

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    # read frames and accumulate sum and sum of squares
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nr += 1
        if frame_nr % sample_step != 0:
            continue

        undist = cv.remap(frame, map1, map2, cv.INTER_LINEAR)
        undist = undist[y:y + h_roi, x:x + w_roi].astype(np.float64)

        sum += undist
        sumsquares += undist * undist
        n += 1

        if max_samples is not None and n >= max_samples:
            break

    cap.release()

    # calculate mean and standard deviation
    mean = (sum / n).astype(np.float32)
    var = (sumsquares / n) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 0.0).astype(np.float32)

    std = np.sqrt(var).astype(np.float32)

    return mean, std, newK, roi, map1, map2


def extract_mask_mahalanobis(frame_bgr, mean, std, T, min_std):
    f = frame_bgr.astype(np.float32)

    # sometimes std is 0 so we take a min std, also for better detection
    sd = np.maximum(std, min_std)
    z = (f - mean) / sd
    d2 = np.sum(z * z, axis=2)
    d = np.sqrt(d2)

    mask = (d > T).astype(np.uint8) * 255
    return mask


def write_mask_video_mahalanobis(video_avi_path, out_avi_path,
                                 mean, std, roi, map1, map2,
                                 T=3.0, min_std=5.0, area=750, aggressiveness=1):
    cap = cv.VideoCapture(video_avi_path)

    if aggressiveness < 0: aggressiveness = 0

    fps = cap.get(cv.CAP_PROP_FPS)
    x, y, w_roi, h_roi = roi

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(out_avi_path, fourcc, fps, (w_roi, h_roi), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undist = cv.remap(frame, map1, map2, cv.INTER_LINEAR)
        undist = undist[y:y + h_roi, x:x + w_roi]

        mask = extract_mask_mahalanobis(undist, mean, std, T=T, min_std=min_std)
        mask = cv.dilate(mask, None, iterations=aggressiveness) # dilation fills gaps, erosion returns to previous size
        maskcp = mask.copy()

        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > area] # dilation
        cv.drawContours(maskcp, contours, -1, (0, 255, 0), -1)
        np.transpose(np.nonzero(maskcp))
        mask -= maskcp
        mask = cv.erode(mask, None, iterations=max(aggressiveness-1, 0)) # erosion REALLY tanks resolution, so by default it's 1-1=0

        # cv.imshow("mask", mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        out.write(mask)
    
    cap.release()
    out.release()


if __name__ == "__main__":
    cv.ocl.setUseOpenCL(False)

    cam_nr = input("Enter camera number: ")

    K, dist = load_intrinsics(f"cam{cam_nr}/intrinsics.xml")

    mean, std, newK, roi, map1, map2 = gaussian_background_model(
        f"cam{cam_nr}/background.avi",
        K, dist,
        sample_step=5,
        max_samples=300
    )

    # Observations:
    # under 15 min_std shows too much of the shadows
    # around 4 and 15 seem to work well for all but 2, left them like this for now
    T = 4.0
    min_std = 15.0

    if cam_nr == 1:
        T = 4.0
        min_std = 15.0

    # more strict for cam 2 because of more noise
    if cam_nr == 2:
        T = 4.0
        min_std = 25.0

    if cam_nr == 3:
        T = 4.0
        min_std = 15.0

    if cam_nr == 4:
        T = 4.0
        min_std = 15.0

    write_mask_video_mahalanobis(
        f"cam{cam_nr}/video.avi",
        f"cam{cam_nr}/mask.avi",
        mean, std,
        roi, map1, map2,
        T,
        min_std
    )

    print("Wrote cam{}/mask.avi".format(cam_nr))