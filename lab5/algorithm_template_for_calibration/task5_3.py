import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from stereo_camera_calibration import map1_left, map2_left, map1_right, map2_right

# Arrays to store object points and image points from all the images.
img_width = 640
img_height = 480

image_dir = os.getenv("REPO_ROOT") + "/lab5/dataset/example/"
number_of_images = 1

calibration_img = cv2.imread(os.getenv("REPO_ROOT") + "/lab5/dataset/pairs/left_01.png")

img = cv2.imread(image_dir + "example0.jpg")
N, XX, YY = img.shape[::-1]
left_img = np.zeros((YY, int(XX / 2), N), np.uint8)
right_img = np.copy(left_img)

left_img = img[:, 0 : int(XX / 2) :, :]
right_img = img[:, int(XX / 2) : XX :, :]

left_img = cv2.resize(
    left_img, (int(calibration_img.shape[1]), int(calibration_img.shape[0]))
)
right_img = cv2.resize(
    right_img, (int(calibration_img.shape[1]), int(calibration_img.shape[0]))
)

left_img_uncalibrated = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
right_img_uncalibrated = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)


left_rectified = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)

left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

block_matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity_bm = block_matcher.compute(left_gray, right_gray)
sgm_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=15)
disparity_sgm = sgm_matcher.compute(left_gray, right_gray)
disparity_bm_norm = cv2.normalize(
    disparity_bm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
)
disparity_sgm_norm = cv2.normalize(
    disparity_sgm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

heatmap_bm = cv2.applyColorMap(disparity_bm_norm, cv2.COLORMAP_MAGMA)
heatmap_sgm = cv2.applyColorMap(disparity_sgm_norm, cv2.COLORMAP_MAGMA)


plt.figure(figsize=(18, 10))

plt.subplot(3, 2, 1)
plt.title("left_img")
plt.imshow(left_img)

plt.subplot(3, 2, 2)
plt.title("right_img")
plt.imshow(right_img)

plt.subplot(3, 2, 3)
plt.title("left_rectified")
plt.imshow(left_rectified)

plt.subplot(3, 2, 4)
plt.title("right_rectified")
plt.imshow(right_rectified)

plt.subplot(3, 2, 5)
plt.title("heatmap_bm")
plt.imshow(heatmap_bm)

plt.subplot(3, 2, 6)
plt.title("heatmap_sgm")
plt.imshow(heatmap_sgm)

plt.show()
