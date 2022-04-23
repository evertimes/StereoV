import cv2

import disparity

disparity = disparity.getDisparityMap("../images/left.png", "../images/right.png", 64, 13, False)
cv2.imwrite("disparity.bmp", disparity)
