import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    img_left = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/left_panorama.jpg")
    img_right = cv2.imread(os.getenv("REPO_ROOT") + "/lab6/source/right_panorama.jpg")
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    siftobject = cv2.SIFT_create()

    kp_left, desc_left = siftobject.detectAndCompute(img_left_gray, None)
    #drawing the keypoints and orientation of the keypoints in the image and then displaying the image as the output on the screen
    kp_img_left = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 0, 255))

    cv2.imshow('SIFT', kp_img_left)
    cv2.waitKey()

    kp_right, desc_right = siftobject.detectAndCompute(img_right_gray, None)
    #drawing the keypoints and orientation of the keypoints in the image and then displaying the image as the output on the screen
    kp_img_right = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 0, 255))
    
    cv2.imshow('SIFT', kp_img_right)
    cv2.waitKey()

