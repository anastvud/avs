import cv2
import numpy as np
import os


image_dir = os.getenv("REPO_ROOT") + "/lab9/"


img = cv2.imread(image_dir + "trybik.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

_, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

img_bin = cv2.bitwise_not(img_gray)


contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2 . drawContours ( img, contours , -1, (0, 255, 0), 3)
# cv2.namedWindow("Resized_Window1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Resized_Window1", 1000, 1000)
# cv2.imshow("Resized_Window1", img)
# cv2.waitKey(0)


sobelx = cv2 . Sobel ( img_gray , cv2 . CV_64F ,1 ,0 , ksize =5)
sobely = cv2 . Sobel ( img_gray , cv2 . CV_64F ,0 ,1 , ksize =5)

gradient_amplitude = np . sqrt ( sobelx **2 + sobely **2)
gradient_amplitude = gradient_amplitude / np.amax(gradient_amplitude)
gradient_orientation = np . arctan2 ( sobely , sobelx) * (180 / np . pi)


moments = cv2.moments(img_bin)
center_x = int(moments['m10'] / moments['m00'])
center_y = int(moments['m01'] / moments['m00'])
reference_point = (center_x, center_y)

Rtable = [[] for _ in range (360) ]

for point in contours[0]:
    x, y = point[0]
    vector = (reference_point[0] - x, reference_point[1] - y)
    distance = np.sqrt(vector[0]**2 + vector[1]**2)
    angle = np.arctan2(vector[1], vector[0]) * (180 / np.pi)
    angle = (angle + 360) % 360
    grad_orientation = int(gradient_orientation[y, x])
    Rtable[grad_orientation].append((distance, angle))


# Load the target image
target_img = cv2.imread(image_dir + "trybiki2.jpg")
target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

# Calculate gradients for the target image
sobelx_target = cv2.Sobel(target_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely_target = cv2.Sobel(target_gray, cv2.CV_64F, 0, 1, ksize=5)

gradient_magnitude_target = np.sqrt(sobelx_target**2 + sobely_target**2)
gradient_magnitude_target = gradient_magnitude_target / np.amax(gradient_magnitude_target)
gradient_orientation_target = np.arctan2(sobely_target, sobelx_target) * (180 / np.pi)

gradient_magnitude_target = gradient_magnitude_target / np.amax(gradient_magnitude_target)

# Initialize Hough space
hough_space = np.zeros(target_gray.shape)


# Fill Hough space
threshold = 0.5
for y in range(target_gray.shape[0]):
    for x in range(target_gray.shape[1]):
        if gradient_magnitude_target[y, x] > threshold:
            orientation = int(gradient_orientation_target[y, x])
            for r, fi in Rtable[orientation]:
                x1 = int(x - r * np.cos(np.deg2rad(fi)))
                y1 = int(y - r * np.sin(np.deg2rad(fi)))
                if 0 <= x1 < target_gray.shape[1] and 0 <= y1 < target_gray.shape[0]:
                    hough_space[y1, x1] += 1


# Find the maximum in Hough space
max_y, max_x = np.unravel_index(np.argmax(hough_space), hough_space.shape)

# Mark the found maximum on the image
cv2.circle(target_img, (max_x, max_y), 5, (0, 0, 255), -1)

# Display the results
cv2.namedWindow("Pattern Contour", cv2.WINDOW_NORMAL)
cv2.imshow("Pattern Contour", img)

cv2.namedWindow("Hough Space", cv2.WINDOW_NORMAL)
cv2.imshow("Hough Space", (hough_space / np.amax(hough_space) * 255).astype(np.uint8))

cv2.namedWindow("Detected Position", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Position", target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.namedWindow("Resized_Window1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Resized_Window1", 1000, 1000)
# cv2.imshow("Resized_Window1", img)
# cv2.waitKey(0)
