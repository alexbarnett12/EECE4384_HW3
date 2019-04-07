import numpy as np
import cv2
# import scipy.linalg import null_space

# filename1 = 'Stonehenge1.png'
# filename2 = 'Stonehenge2.png'
# filename1 = './calibPictures/opencv_frame_0.png'
# filename2 = './calibPictures/opencv_frame_1.png'

filename1 = 'classroom1.png'
filename2 = 'classroom2.png'

def orb_detector(img):

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints with ORB
    kp = orb.detect(img, None)

    # Compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # Draw keypoints location
    img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    return kp, des, img


def feature_matching(norm, img1, img2, kp1, kp2, des1, des2, numMatches, display, fileout):

    # create BFMatcher object
    bf = cv2.BFMatcher(norm, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    # img3 = np.array([])
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:numMatches], img3, flags=2)
    #
    # cv2.imshow(display, img3)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imwrite('./images/feature_match/' + fileout, img3)

    return matches


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, B = img1.shape
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        # img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        # img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find good conjugate pixel pairs from feature matching on a stereo pair of images

''' ORB Detector '''
img_left = cv2.imread(filename1)
img_right = cv2.imread(filename2)
R, C, B = np.shape(img_left)

kp_left, des_left, img_left_orb = orb_detector(img_left)
kp_right, des_right, img_right_orb = orb_detector(img_right)

# Visualize ORB detector
# cv2.imshow('ORB Corner Detector 1', img1_orb)
# cv2.imshow('ORB Corner Detector 2', img2_orb)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
#
# Write images to files
# cv2.imwrite('./results/orb/orb1.png', img1_orb)
# cv2.imwrite('./results/orb/orb2.png', img2_orb)

''' ORB Feature Matching '''
matches = feature_matching(cv2.NORM_HAMMING, img_left, img_right, kp_left, kp_right, des_left, des_right, 100,
                           'ORB Feature Matching', 'featureMatch_orb.png')

# Load camera parameters
calib = np.load('./data/calib.npz')
parameters = calib._files
Kn, distn, rvecn, tvecn = [parameters[0], parameters[1], parameters[2], parameters[3]]
cameraMatrix, distCoeffs, rvec, tvec = [calib[parameters[0]], calib[parameters[1]],
                                        calib[parameters[2]], calib[parameters[3]]]

# Undistort images
img_left_undistorted = cv2.undistort(img_left, cameraMatrix, distCoeffs)
img_right_undistorted = cv2.undistort(img_right, cameraMatrix, distCoeffs)

# Display undistorted images
# cv2.imshow('Undistorted Image Left', img_left_undistorted)
# cv2.imshow('Undistorted Image Right', img_right_undistorted)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

newK, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (C, R), 0, (C, R))

# N x 1 x 2
matched_kps_left = [kp_left[mat.queryIdx].pt for mat in matches]
matched_kps_right = [kp_right[mat.trainIdx].pt for mat in matches]
matched_kps_left = np.array(matched_kps_left)
matched_kps_right = np.array(matched_kps_right)

matched_kps_left = matched_kps_left[:, np.newaxis, :]
matched_kps_right = matched_kps_right[:, np.newaxis, :]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# Undistort the image points
pts_left_undistorted = cv2.undistortPointsIter(matched_kps_left, cameraMatrix, distCoeffs,
                                               R=None, P=newK, criteria=criteria)
pts_right_undistorted = cv2.undistortPointsIter(matched_kps_right, cameraMatrix, distCoeffs,
                                                R=None, P=newK, criteria=criteria)

# Compute the fundamental matrix
F, mask = cv2.findFundamentalMat(pts_left_undistorted, pts_right_undistorted,
                                 method=cv2.RANSAC, ransacReprojThreshold=0.5, confidence=.99)

# Compute the essential matrix
E = np.matmul(np.transpose(cameraMatrix), np.matmul(F, cameraMatrix))

# Recover inliers
inl_pts_left = matched_kps_left[mask.ravel() == 1]
inl_pts_right = matched_kps_right[mask.ravel() == 1]
nPts = inl_pts_left.shape[0]

newPts_left, newPts_right = cv2.correctMatches(F, np.reshape(inl_pts_left, (1, nPts, 2)),
                                               np.reshape(inl_pts_right, (1, nPts, 2)))
newPts_left = newPts_left.reshape(-1, 1, 2)
newPts_right = newPts_right.reshape(-1, 1, 2)

# Draw epilines on rectified image
lines1 = cv2.computeCorrespondEpilines(newPts_right.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img_left, img_right, lines1, newPts_left, newPts_right)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(newPts_left.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img_left, img_right, lines2, newPts_left, newPts_right)
# cv2.imshow('Left Image w/ epilines', img5)
# cv2.imshow('Right Image w/ epilines', img3)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

# Decompose essential matrix into translational and rotational vectors
ret, rot, trans, mask, wPts = cv2.recoverPose(E, newPts_left, newPts_right, newK, distanceThresh=1000)
np.savez('./data/FMat.npz', F=F, E=E, rvec=rot, tvec=trans)

# Homogenize worldpoint estimates
wPts = cv2.convertPointsFromHomogeneous(np.transpose(wPts))

# Calculate left and right projection matrices
PL = np.append(newK, np.zeros((3, 1), dtype=float), axis=1)
PR = np.append(newK, np.append(rot, trans, axis=1))
invRot = np.transpose(rot)
invTrans = -np.matmul(invRot, trans)

# Alternate projection matrix; not needed for assignment
# eR = linalg.null_space(np.transpose(F))
# eRmat = np.skew(eR)
# PR_alt = np.matmul(newK, np.append((np.matmul(eRmat, F) + np.matmul(eR, np.transpose(eR))), eR, axis=1))

# Compute rotation vector from matrix
rotVec = cv2.Rodrigues(rot)

# Project triangulated world points onto the left image plane
proj_left = cv2.projectPoints(wPts, (0, 0, 0), (0, 0, 0), newK, distCoeffs)
# Project triangulated world points onto the right image plane
proj_right = cv2.projectPoints(wPts, rotVec[0], trans, newK, distCoeffs)
np.savez('./data/ptbased', rotVec=rotVec, trans=trans)

# Compute reprojection error
# Compare reprojected points to undistorted original points
# by computing average Euclidian norm of the difference

# Distance between points in all axes
dLx = np.squeeze(proj_left[0][:, 0, 0] - newPts_left[:, 0, 0])
dRx = np.squeeze(proj_right[0][:, 0, 0] - newPts_right[:, 0, 0])
dLy = np.squeeze(proj_left[0][:, 0, 1] - newPts_left[:, 0, 1])
dRy = np.squeeze(proj_right[0][:, 0, 1] - newPts_right[:, 0, 1])

# Error
pt_error_left_x = np.sum(np.abs(dLx))/nPts
pt_error_right_x = np.sum(np.abs(dRx))/nPts
pt_error_left_y = np.sum(np.abs(dLy))/nPts
pt_error_right_y= np.sum(np.abs(dRy))/nPts

# Compute homography matrix
ret, HL, HR = cv2.stereoRectifyUncalibrated(newPts_left, newPts_right, F, (C, R), threshold=1)

# Rectify images
img_left_rect = cv2.warpPerspective(img_left_undistorted, HL, (C, R))
img_right_rect = cv2.warpPerspective(img_right_undistorted, HR, (C, R))

# Display rectified images
# cv2.imshow('Rectified Left Image', img_left_rect)
# cv2.imshow('Rectified Right Image', img_right_rect)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
cv2.imwrite('./images/rect_left.png', img_left_rect)
cv2.imwrite('./images/rect_right.png', img_right_rect)


gray_left_rect = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
gray_right_rect = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)

stereoMatcher = cv2.StereoBM_create(256, 25)
disparity = stereoMatcher.compute(gray_left_rect, gray_right_rect)
disparity = cv2.normalize(src=disparity, dst=disparity, beta=-16, alpha=255, norm_type=cv2.NORM_MINMAX)
# disp_color = cv2.applyColorMap(np.uint8(disparity), 2)

# disparity = stereoMatcher.compute(img_left_rect, img_right_rect)
# disparity = stereoMatcher.compute(thresh_left, thresh_right)

cv2.imshow('Rectified Right Image', disparity)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# Class Notes 4/1
''' Use cv.undistort to undistort image DONE
    Use cv.undistortPointsIter(kp, newK, dist, R=None,  to undistort keypoints (input keypoints and camera matrix stuff)
    Do this since you need to pass undistroted points into fundamental matrix method DONE
    then use cv.findFundamentalMatrix
        RANSAC as a procedure: use a very high confidence level (~.99) and low re=projection threshold (~0.5) DONE 
    Comes back with F, mask. mask are the inlier points, so you want to save those as separate variables DONE
    Then compute the essential matrix by premultiplying by camera matrix on both sides (look at slides) DONE
    Decompose essential matrix into translational and rotational vectors DONE
    Make a unit vector out of the translation vector if it isn't already (divide by L2 norm) DONE 
    
    
    ///
    Then multiply it by a number which is roughly the same as the distance between the two pictures taken 
    Use triangulation on inlier points to make 3d points (opencv func?)
    Then take 3d vectors and "run them through" left and right camera matrices
    Then compute left and right homography matrices (opencv funcs)
        ret, HL, HR = cv.stereoRectifyUncalibrated
    HL /= HL[2,2] DONT NEED THESE FOUR LINES
    HR /= HR[2,2]
    HL[0,2]-=150 # subtract # of pixels from third column to shift things over 
    HR[0,2]-=150
    save these images

    rect_img_left = cv.warpPerspective(undist_img_left, HL, (C, R))

    At this point, you should be able to go straight across images when viewing and have them both be aligned

    Now do stereo matching
    Block Matcher or Peter's program
    http://timosam.com/python_opencv_depthimage
        will need opencv_+contrib thingy
        cv.ximgproc will execute even though it will say it doesn't know what it is

    '''

# # FROM CLASS
#
# newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, {C, R}, 0, {C, R})
#
# # project points through inverse camera matrix and undistort them
# cv2.undistortPointsIter(matched_kps_left, newK, dist,)
#
# F, mask = cv2.findFundamentalMat(undist_pts_left, undist_pts_right, cv2.FM_RANSAC, ransacReprojThreshold=0.5, confidence=)
# np.savez('FMat.npz',F=F)
#
# in1_pts_left = matched_kps_left(mask.ravel()==1)
# in1_pts_right = matched_kps_right(mask.ravel()==1)
#
# newPts_left, newPts_right = cv2.correctMatches(F, np.reshape(in1_pts_left, (1, nPts))
#
# # Find epilines corresponding to points in right image and drawing its lines on the second image
# eLines_left = cv2.computeCorrespondEpilines(newPts_right, 2, F))
# eLines_left = eLines_left.reshape()
#
# ret, HL, HR = cv2.stereoRectifyUncalibrated(newPts_left[:, :, 0:2], newPts_right[:, :, 0:2})
# HL /= HL[2,2]
# HR /= HR[2,2]
# HL[0,2]-= 150
# HR[0,2] -= 150
#
# rect_img_left = cv2.warpPerspective(undist_img_left, HL, (C, R))
# rect_img_right = cv2.warpPerspective(undist_img_right, HR, (C, R))
#
# #Convert to grayscale
#
# #Now use StereoBM
# stereoMatcher = cv2.StereoBM_create()
# stereoMatcher.setMinDisparity(16)
# stereoMatcher.setBlockSize(9)
# disparity = stereoMatcher.compute(gray_left, gray_right)
# disparity_norm = cv2.normalize(disparity, None, 255, 0,cv2.NORM_MINMAX, cv2.CV_8UC1)
# plt.imshow(disparity,'gray')
# plt.show()


