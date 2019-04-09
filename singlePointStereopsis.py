# Single Point Stereopsis
# Alex Barnett, Spring 2019
# Code translated from Dr. Peters' Matlab code

import numpy as np


# Haralick and Shapiro depth calculator
#
# Assumes image metric coords with focal length as z-component, or normalized components with z = 1
# Rotations and translation are wrt the cameras' coordinate systems and transform camera coordinates into world
# coordinates
def hs_stereo(pt_c_l, rvec_l, tvec_l, pt_c_r, rvec_r, tvec_r):
    # Rotate the image points into alignment with the world coordinates
    pwl = np.matmul(rvec_l, pt_c_l)
    pwr = np.matmul(rvec_r, pt_c_r)

    # Construct a covariance matrix between pwl and pwr

    # Magnitude squared of world points
    pl2 = np.asscalar(np.matmul(np.transpose(pwl), pwl))
    pr2 = np.asscalar(np.matmul(np.transpose(pwr), pwr))

    # Covariance of right point with left
    plr = np.asscalar(np.matmul(np.transpose(pwl), pwr))

    # Covariance of left point with right
    prl = np.asscalar(np.matmul(np.transpose(pwr), pwl))

    C = np.array([[pl2, prl], [plr, pr2]])

    # Calculate the determinant
    k = 1 / (pl2 * pr2 - plr * prl)

    # Matrix inverse
    Cadj = np.array([[pr2, -prl], [-plr, pl2]])
    Cinv = k * Cadj

    # Sign change of second row
    Cinv = np.matmul(Cinv, np.array([[1, 0], [0, -1]]))

    # Compute the baseline
    base = tvec_r - tvec_l

    # Project each point onto the baseline
    b1 = np.asscalar(np.matmul(np.transpose(base), pwl))
    b2 = np.asscalar(np.matmul(np.transpose(base), pwr))

    # Transform the baseline projections into the pwl-pwr cdt sys
    baselines = np.array([[b1], [b2]])
    lamb = np.matmul(Cinv, baselines)

    # Re-express each in terms of world coordinates
    # rvec_r = np.reshape(rvec[1][0], (3,1))
    pwlest = np.asscalar(lamb[0])* pwl + tvec_l
    pwrest = np.asscalar(lamb[1]) * pwr + tvec_r

    # Take their average
    pw = (pwlest + pwrest) / 2

    return pw

# Load camera parameters
calib = np.load('./data/FMat.npz')
parameters = calib._files
rvec, tvec = [calib[parameters[2]], calib[parameters[0]]]

# Visually select four point pairs that are on objects at different distances

pl = np.array([[279], [428], [1]])
pr = np.array([[312], [430], [1]])


# Run HSstereo algorithm
rvec_l = np.identity(3)
tvec_l = np.array([[0], [0], [0]])

pw = hs_stereo(pl, rvec_l, tvec_l, pr, rvec, tvec)
print(pw)
