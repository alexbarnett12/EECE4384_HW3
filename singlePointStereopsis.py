import numpy as np
import cv2

# Import essential matrix and camera intrinsic parameters

# Visually select four point pairs that are on objects at different distances

pL = np.array([1,1,1])
pR = np.array([1,1,1])
p = np.array([pL, pR])

# Rotate camera coords of point into world frame
q_lw = R_lw*pL
q_rw = R_rw*pR
q = np.array([q_lw, q_rw])

# Compute baseline and project onto rotated image vectors
T_w = np.array([T_lw, T_rw])
T_rl = T[1] - T[0]
T_lr = T[0] - T[1]


bL = cv2.norm(q_lw)*cv2.norm(T_rl)*np.cos(theta_lt)
bR = cv2.norm(q_rw)*cv2.norm(T_lr)*np.cos(theta_rt)
b = np.array([bL, bR])

# Calculate covariance matrix
cov = np.cov(q)

# Take the inverse of the covariance matrix
cov_inv = np.linalg.inv(cov)

# Multiply the inverse by baseline vector
inverse_proj = cov_inv*b

# Find two pixel depth estimates and take the average
p_lw = inverse_proj[0] * q[0] + T_w[0]
p_rw = inverse_proj[1] * q[1] + T_w[1]