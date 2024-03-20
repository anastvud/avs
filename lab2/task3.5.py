import cv2
import numpy as np


def calculate_metrics(TP, TN, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0
    )
    return precision, recall, f1_score


def calculate_median_mean(buffer):
    median = np.median(buffer, axis=2).astype(np.uint8)
    mean = np.mean(buffer, axis=2).astype(np.uint8)
    return median, mean


def binarization(img):
    _, img = cv2.threshold(img, 18, 255, cv2.THRESH_BINARY)

    img = cv2.medianBlur(img, 5)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    img = cv2.medianBlur(img, 9)
    return img


def calculate_parameters(img, ground_truth, TP, TN, FP, FN):
    TP_M = np.logical_and((img == 255), (ground_truth == 255))
    TP_S = np.sum(TP_M)
    TP += TP_S

    FP_M = np.logical_and((img == 255), (ground_truth == 0))
    FP_S = np.sum(FP_M)
    FP += FP_S

    FN_M = np.logical_and((img == 0), (ground_truth == 255))
    FN_S = np.sum(FN_M)
    FN += FN_S

    TN_M = np.logical_and((img == 0), (ground_truth == 0))
    TN_S = np.sum(TN_M)
    TN += TN_S

    return TP, TN, FP, FN


prev = cv2.imread("highway/input/in000300.jpg")
N = 60 
BUF = np.zeros((prev.shape[0], prev.shape[1], N), np.uint8)
iN = 0
check = False
median = mean = None
TP, TN, FP, FN= 0, 0, 0, 0
background_model_median = background_model_mean = None
alpha = 0.03

backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=500, detectShadows=False)


for i in range(300, 1100):
    curr = cv2.imread("highway/input/in%06d.jpg" % i)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    fg_mask = backSub.apply(curr_gray)

    ground_truth_mask = cv2.imread("highway/groundtruth/gt%06d.png" % i)
    ground_truth_mask = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)


    TP, TN ,FP, FN = calculate_parameters(
        fg_mask, ground_truth_mask, TP, TN, FP, FN
    )


    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original", 1000, 1000)
    cv2.imshow("original", curr_gray)
    cv2.waitKey(10)

    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask", 1000, 1000)
    cv2.imshow("mask", fg_mask)
    cv2.waitKey(10)

    prev = curr

precision, recall, f1_score = calculate_metrics(TP, TN, FP, FN)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
