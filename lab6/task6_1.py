import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter


def Harris(image, sobel_size, gauss_size):
    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=sobel_size)
    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=sobel_size)

    xx = cv2.GaussianBlur(x * x, (gauss_size, gauss_size), 0)
    yy = cv2.GaussianBlur(y * y, (gauss_size, gauss_size), 0)
    xy = cv2.GaussianBlur(x * y, (gauss_size, gauss_size), 0)

    K = 0.05
    det = xx * yy - xy ** 2
    trace = xx + yy
    H = det - K * trace ** 2

    H = cv2.normalize(
        H, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    return H


def find_max(image, size, threshold):
    data_max = maximum_filter(image, size)
    maxima = image == data_max
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


def draw_marks(image, coordinates):
    plt.figure()
    plt.imshow(image)
    for coord in zip(*coordinates):
        plt.plot(
            coord[1], coord[0], "*", color="r"
        )  # Swap x and y coordinates for plotting
    plt.show()


if __name__ == "__main__":

    sobel_filter = gauss_filter = 7
    threshold = 0.5

    img_1 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/fontanna1.jpg")
    img_2 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/fontanna2.jpg")
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    H1 = Harris(img_1_gray, sobel_filter, gauss_filter)
    H2 = Harris(img_2_gray, sobel_filter, gauss_filter)

    corners1 = find_max(H1, sobel_filter, threshold)
    corners2 = find_max(H2, sobel_filter, threshold)
    draw_marks(img_1, corners1)
    draw_marks(img_2, corners2)


    img_3 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/budynek1.jpg")
    img_4 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/budynek2.jpg")
    img_3_gray = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
    img_4_gray = cv2.cvtColor(img_4, cv2.COLOR_BGR2GRAY)

    H3 = Harris(img_3_gray, sobel_filter, gauss_filter)
    H4 = Harris(img_4_gray, sobel_filter, gauss_filter)

    corners3 = find_max(H3, sobel_filter, threshold)
    corners4 = find_max(H4, sobel_filter, threshold)
    draw_marks(img_3, corners3)
    draw_marks(img_4, corners4)
   