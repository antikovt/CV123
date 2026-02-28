import cv2 as cv
import numpy as np

# drops inaccurate images based on reprojection error
def reprojection_error_filter(objpoints, imgpoints, image_names,
                                     mtx, dist, rvecs, tvecs,
                                     threshold_px):
    per_image_error = []
    mean_error = 0

    # compute reprojection error for each image and the mean error across all images
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
        per_image_error.append(error)

    mean_error /= len(objpoints)

    print("\nReprojection errors per image:")
    for name, err in zip(image_names, per_image_error):
        print(f"{name}: {err:.3f} px")

    print(f"\nMean reprojection error: {mean_error:.3f} px")


    keep_idx = []
    drop_idx = []

    # drop images with reprojection error above threshold
    for i in range(len(per_image_error)):
        error = per_image_error[i]
        if error <= threshold_px:
            keep_idx.append(i)
        else:
            drop_idx.append(i)

    print("\nDropping images:")
    for i in drop_idx:
        print(image_names[i])

    obj_keep = [objpoints[i] for i in keep_idx]
    img_keep = [imgpoints[i] for i in keep_idx]
    names_keep = [image_names[i] for i in keep_idx]

    return obj_keep, img_keep, names_keep


# Same function but no report. Filters quietly
def reprojection_error_filter_silent(objpoints, imgpoints,
                                     mtx, dist, rvecs, tvecs,
                                     threshold_px):
    per_image_error = []
    mean_error = 0

    # compute reprojection error for each image and the mean error across all images
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
        per_image_error.append(error)

    keep_idx = []

    # drop images with reprojection error above threshold
    for i in range(len(per_image_error)):
        error = per_image_error[i]
        if error <= threshold_px:
            keep_idx.append(i)

    obj_keep = [objpoints[i] for i in keep_idx]
    img_keep = [imgpoints[i] for i in keep_idx]

    return obj_keep, img_keep
