import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from source.pm import plot_matches

import task6_1


def get_patches(image, points, patch_size):
    X, Y = image.shape
    half_size = patch_size // 2
    filtered_points = list(
        filter(
            lambda pt: pt[0] >= half_size
            and pt[0] < Y - half_size
            and pt[1] >= half_size
            and pt[1] < X - half_size,
            zip(points[0], points[1]),
        )
    )
    patches = []
    for pt in filtered_points:
        patch = image[pt[1]-half_size:pt[1]+half_size+1, pt[0]-half_size:pt[0]+half_size+1]
        mean = np.mean(patch)
        std = np.std(patch)
        normalized_patch = (patch - mean) / std
        patches.append(normalized_patch.flatten())
    return list(zip(patches, filtered_points))


def compare_descriptions(desc1, desc2, n):
    matches = []
    for d1, pt1 in desc1:
        similarities = [(d2, pt2, np.linalg.norm(d1 - d2)) for d2, pt2 in desc2]
        similarities.sort(key=lambda x: x[2], reverse=True)
        matches.append((pt1, similarities[0][1], similarities[0][2]))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:n]


if __name__ == "__main__":
    sobel_filter = gauss_filter = 7
    threshold = 0.5
    patch_size = 20 
    img_1 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/fontanna1.jpg")
    img_2 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/fontanna2.jpg")
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    H1 = task6_1.Harris(img_1_gray, sobel_filter, gauss_filter)
    H2 = task6_1.Harris(img_2_gray, sobel_filter, gauss_filter)

    corners1 = task6_1.find_max(H1, sobel_filter, threshold)
    corners2 = task6_1.find_max(H2, sobel_filter, threshold)
    # task6_1.draw_marks(img_1, corners1)
    # task6_1.draw_marks(img_2, corners2)

    patches1 = get_patches(img_1_gray, corners1, patch_size)
    patches2 = get_patches(img_2_gray, corners2, patch_size)

    matches = compare_descriptions(patches1, patches2, 10)
    plot_matches(img_1_gray, img_2_gray, matches)

    img_3 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/budynek1.jpg")
    img_4 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/budynek2.jpg")
    img_3_gray = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
    img_4_gray = cv2.cvtColor(img_4, cv2.COLOR_BGR2GRAY)

    H3 = task6_1.Harris(img_3_gray, sobel_filter, gauss_filter)
    H4 = task6_1.Harris(img_4_gray, sobel_filter, gauss_filter)

    corners3 = task6_1.find_max(H3, sobel_filter, threshold)
    corners4 = task6_1.find_max(H4, sobel_filter, threshold)
    # task6_1.draw_marks(img_3, corners3)
    # task6_1.draw_marks(img_4, corners4)

    patches3 = get_patches(img_3_gray, corners3, patch_size)
    patches4 = get_patches(img_4_gray, corners4, patch_size)

    matches = compare_descriptions(patches3, patches4, 10)
    plot_matches(img_3_gray, img_4_gray, matches)
