import numpy as np
import cv2
from matplotlib import pyplot as plt


imgL = cv2.imread('classroom1.png', 0)
imgR = cv2.imread('classroom2.png', 0)

imgLG = cv2.GaussianBlur(imgL,(11,11),0)
imgRG = cv2.GaussianBlur(imgR,(11,11),0)

stereoMatcher = cv2.StereoBM_create(256, 25)

# stereoMatcher = cv2.StereoBM_create()
# stereoMatcher.setMinDisparity(256)
# stereoMatcher.setBlockSize(25)
disparity = stereoMatcher.compute(imgL, imgR)
disparityG = stereoMatcher.compute(imgLG, imgRG)
plt.imshow(disparity,'gray')
plt.show()

cv2.imwrite('./images/BM_disp.png',disparity)
cv2.imwrite('./images/BM_disp_GB.png',disparityG)
