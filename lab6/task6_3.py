import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import task6_1
import pandas as pd
from source.pm import plot_matches




def fast(image, patch_size=31, n=256) -> list:
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(image,None)

    half_size = patch_size // 2
    height, width = image.shape
    filtered_kp = [k for k in kp if half_size <= k.pt[0] < width - half_size and half_size <= k.pt[1] < height - half_size]
    
    sorted_kp = sorted(filtered_kp, key=lambda x: x.response, reverse=True)
    sorted_kp = sorted_kp[:n]
    coords_list = [(kp.pt[0], kp.pt[1]) for kp in sorted_kp]
    # print(coords_list)

    patches = []
    for pt in coords_list:
        patch = image[int(pt[1])-half_size:int(pt[1])+half_size+1, int(pt[0])-half_size:int(pt[0])+half_size+1]
        mean = np.mean(patch)
        std = np.std(patch)
        normalized_patch = (patch - mean) / std
        patches.append(normalized_patch.flatten())

    centroids = centroid_data(image, coords_list, patch_size)

    return list(zip(patches, coords_list, centroids))

def centroid_data(image, keypoints, patch_size=31) -> list:
    centroids = []
    half_size = patch_size // 2
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        patch = image[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
        
        m00 = np.sum(patch)
        m10 = np.sum(np.arange(-half_size, half_size+1) * np.sum(patch, axis=0))
        m01 = np.sum(np.arange(-half_size, half_size+1)[:, np.newaxis] * np.sum(patch, axis=1))
        
        centroid_x = m10 / m00
        centroid_y = m01 / m00
        theta = np.arctan2(centroid_y, centroid_x)
        centroids.append((centroid_x, centroid_y, theta))
    return centroids

def brief(fast_result, image, patch_size=31) -> None:
    df = pd.read_csv(os.getenv("REPO_ROOT") + "/lab6/source/orb_descriptor_positions.txt", sep=' ', header=None)
    pairs = df.to_numpy()
    descriptors = []

    half_size = patch_size // 2

    for patch, keypoint, centroid in fast_result:
        x, y = keypoint
        x = int(x)
        y = int(y)
        area = image[x - half_size // 2:x + half_size // 2 + 1, y - half_size // 2:y + half_size // 2 + 1]
        area = cv2.GaussianBlur(area, (5, 5), 2.1)

        descriptor = np.zeros(pairs.shape[0], dtype=np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(pairs):
            x1_rot, y1_rot = rotate_point(x1, y1, centroid[2])
            x2_rot, y2_rot = rotate_point(x2, y2, centroid[2])
            descriptor[i] = np.array(area[int(x1_rot), int(y1_rot)] < area[int(x2_rot), int(y2_rot)])

        descriptors.append((keypoint, descriptor))
    
    return descriptors
    
def match_descriptors(desc1, desc2, max_distance=30, n=10):
    matches = []
    keypoints1, descriptors1 = zip(*desc1)
    keypoints2, descriptors2 = zip(*desc2)
    descriptors1 = np.array(descriptors1)
    descriptors2 = np.array(descriptors2)

    for i, d1 in enumerate(descriptors1):
        distances = np.logical_xor(d1, descriptors2).sum(axis=1)
        closest = np.argmin(distances)
        if distances[closest] <= max_distance:
            matches.append((keypoints1[i], keypoints2[closest], distances[closest]))

    matches.sort(key=lambda x: x[2])
    matches = matches[:n]

    return [((int(x1), int(y1)), (int(x2), int(y2))) for ((x1, y1), (x2, y2), _) in matches]     


def rotate_point(x, y, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * y + cos_theta * x
    return x_rot, y_rot


if __name__ == "__main__":
    sobel_filter = gauss_filter = 7
    threshold = 0.5
    patch_size = 20 
    img_1 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/fontanna1.jpg")
    img_2 = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/fontanna2.jpg")
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)


    patches_1 = fast(img_1_gray)
    patches_2 = fast(img_2_gray)

    desc_1 = brief(patches_1, img_1_gray)
    desc_2 = brief(patches_2, img_2_gray)
    matches = match_descriptors(desc_1, desc_2)
    plot_matches(img_1, img_2, matches)
    
    