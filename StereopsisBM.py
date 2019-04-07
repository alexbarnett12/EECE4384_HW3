import numpy as np
import cv2
from matplotlib import pyplot as plt

# imgL = cv2.imread('./calibPictures/Stonehenge1.png', 0)
# imgR = cv2.imread('./calibPictures/Stonehenge2.png', 0)
imgL = cv2.imread('classroom1.png', 0)
imgR = cv2.imread('classroom2.png', 0)
# gray_left = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
# gray_right = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

stereoMatcher = cv2.StereoBM_create(256, 25)

# stereoMatcher = cv2.StereoBM_create()
# stereoMatcher.setMinDisparity(256)
# stereoMatcher.setBlockSize(25)
disparity = stereoMatcher.compute(imgL, imgR)
plt.imshow(disparity,'gray')
plt.show()
