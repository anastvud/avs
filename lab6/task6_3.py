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

def brief(fast_result, patch_size=31) -> None:
    df = pd.read_csv(os.getenv("REPO_ROOT") + "/lab6/source/orb_descriptor_positions.txt", sep=' ', header=None)
    pairs = df.to_numpy()
    descriptor = np.zeros(pairs.size, dtype=np.uint8)
    half_size = patch_size // 2

    for i, (patch, keypoint, centroid), (x1, y1, x2, y2) in enumerate(fast_result), pairs:
        x1_rot, y1_rot = rotate_point(x1, y1, centroid)
        x2_rot, y2_rot = rotate_point(x2, y2, centroid)
        if patch[int(y1_rot)+half_size, int(x1_rot)+half_size] < patch[int(y2_rot)+half_size, int(x2_rot)+half_size]:
            descriptor[i] = 1
    
    return descriptor
    
def match_descriptors(desc1, desc2, max_distance=30):
    matches = []
    for (pt1, d1) in desc1:
        best_match = None
        best_distance = max_distance
        for (pt2, d2) in desc2:
            distance = np.sum(d1 != d2)
            if distance < best_distance:
                best_distance = distance
                best_match = (pt1, pt2)
        if best_match:
            matches.append(best_match)
    return matches        


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

    desc_1 = brief(patches_1)
    desc_2 = brief(patches_2)
    matches = match_descriptors(desc_1, desc_2)
    pm.plot_matches(img_1, img_2, matches)
    
    