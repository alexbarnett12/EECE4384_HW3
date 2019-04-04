import numpy as np
import cv2

# Haralick and Shapiro depth calculator
#
# Assumes image metric coords with focal length as z-component, or normalized components with z = 1
# Rotations and translation are wrt the cameras' coordinate systems and transform camera coordinates into world
# coordinates
def HSstereo(pt_c_l, rvec_l, tvec_l, pt_c_r, rvec_r, tvec_r):
    # Rotate the image points into alignment with the world coordinates
    pwl = np.matmul(rvec_l, pt_c_l)
    pwr = np.matmul(rvec_r, pt_c_r)

    # Construct a covariance matrix between pwl and pwr

    # Magnitude squared of world points
    pl2 = np.matmul(np.transpose(pwl), pwl)
    pr2 = np.matmul(np.transpose(pwr), pwr)

    # Covariance of right point with left
    plr = np.matmul(np.transpose(pwl), pwr)

    # Covariance of left point with right
    prl = np.matmul(np.transpose(pwr), pwl)

    C = np.array([pl2, prl], [plr, pr2])

    # Calculate the determinant
    k = 1 / (np.matmul(pl2,pr2) - np.matmul(plr,prl))

    # Matrix inverse
    Cadj = np.array([pr2, -prl], [-plr, pl2])
    Cinv = k * Cadj

    # Sign change of second row
    Cinv = Cinv * np.array([1, 0], [0, -1])

    # Compute the baseline
    base = tvec_r - tvec_l

    # Project each point onto the baseline
    b1 = np.matmul(np.transpose(base),pwl)
    b2 = np.matmul(np.transpose(base),pwr)

    # Transform the baseline projections into the pwl-pwr cdt sys
    lamb = Cinv * np.array([b1, b2])

    # Re-express each in terms of world coordinates
    pwlest = np.matmul(pwl, lamb[0]) + tvec_l
    pwrest = np.matmul(pwr, lamb[1]) + tvec_r

    # Take their average
    pw = (pwlest + pwrest) / 2

    return pw


# Import essential matrix and camera intrinsic parameters
calib = np.load('./data/calib.npz')
parameters = calib._files
# K, dist, rvec, tvec = [calib[parameters[0]], calib[parameters[1]],
#                                         calib[parameters[2]], calib[parameters[3]]]
K, dist= [calib[parameters[0]], calib[parameters[1]]]

# Import rvec and tvec extraced from essential matrix
projection_parameters = np.load('./data/FMat.npz')
parameters = projection_parameters._files
F, E, rvec, tvec = [calib[parameters[0]], calib[parameters[1]],calib[parameters[2]], calib[parameters[3]]]


# Visually select four point pairs that are on objects at different distances
