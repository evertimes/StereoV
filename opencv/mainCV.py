import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('../images/im0.png',0)
imgR = cv.imread('../images/im1.png',0)
stereo = cv.StereoSGBM_create(numDisparities=32, blockSize=17)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.show()