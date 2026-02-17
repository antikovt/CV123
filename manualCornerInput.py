import cv2 as cv
import numpy as np

# manual input of corners by clicking on the image
def click_event(event, x, y, flags, param):
    offset = 5 # offset to position O closer to the click point
    if event == cv.EVENT_LBUTTONDOWN:
        param["corners"].append([x, y])
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(param["img2"], "O", (x - offset, y + offset), font, 0.5, (255, 0, 0), 2)
        cv.imshow('image', param["img2"])

# manual input of corners, takkes the image and outputs the corners array
def manual_corner_input(img):

    img2 = img.copy()
    corners = []
    param = {"img2": img2, "corners": corners}

    cv.imshow("image", img2)
    cv.setMouseCallback("image", click_event, param)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # findChessboardCoarners returns np.array of type np.float32, needed for cornerSubPix
    corners = np.array(corners, np.float32)
    # Make return true again because now we have corners
    return True, np.array(corners, np.float32)

