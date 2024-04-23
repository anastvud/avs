import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from stereo_camera_calibration import map1_left, map2_left, map1_right, map2_right


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

# inner size of chessboard
width = 9
height = 6
square_size = 0.025  # 0.025 meters

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((height * width, 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

objp = objp * square_size  # Create real world coords. Use your metric.

# Arrays to store object points and image points from all the images.
img_width = 640
img_height = 480
image_size = (img_width, img_height)


image_dir = os.getenv("REPO_ROOT") + "/lab5/dataset/example/"
number_of_images = 1

img = cv2.imread(image_dir + "example0.jpg")
N, XX, YY = img.shape[::-1]
left_img = np.zeros((YY, int(XX / 2), N), np.uint8)
right_img = np.copy(left_img)

left_img = img[:, 0:int(XX / 2):, :]
right_img = img[:, int(XX / 2) : XX :, :]

right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)



# map1_left_copy = np.copy(map1_left)
# map2_left_copy = np.copy(map2_left)
# map1_right_copy = np.copy(map1_right)
# map2_right_copy = np.copy(map2_right)

# map1_left_copy.resize(int(map1_left.shape[0] / 4), int(map1_left.shape[1] / 4)).astype(np.int16)
# map2_left_copy.resize(int(map2_left.shape[0] / 4), int(map2_left.shape[1] / 4)).astype(np.int16)
# map1_right_copy.resize(int(map1_right.shape[0] / 4), int(map1_right.shape[1] / 4)).astype(np.int16)
# map2_right_copy.resize(int(map2_right.shape[0] / 4), int(map2_right.shape[1] / 4)).astype(np.int16)     

# print("Type of map1: ", map1_left_copy.dtype)
# print("Type of map2: ", map2_left_copy.dtype)
# print("Type of map1: ", map1_right_copy.dtype)
# print("Type of map2: ", map2_right_copy.dtype)

# dst_L = cv2.remap(left_gray, map1_left_copy, map2_left_copy, cv2.INTER_LINEAR)
# dst_R = cv2.remap(right_gray, map1_right_copy, map2_right_copy , cv2.INTER_LINEAR)
# # print(wfjskjfs.shape)

# cv2.imshow('dst_L',dst_L)  # display image with lines
# cv2.imshow('dst_R',dst_R)  # display image with lines
# cv2.waitKey(0)



