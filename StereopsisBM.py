import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('./calibPictures/Stonehenge1.png', 0)
imgR = cv2.imread('./calibPictures/Stonehenge2.png', 0)

stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(16)
stereoMatcher.setBlockSize(15)
disparity = stereoMatcher.compute(imgL, imgR)
plt.imshow(disparity,'gray')
plt.show()
