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
N, iN = 60, 0
BUF = np.zeros((prev.shape[0], prev.shape[1], N), np.uint8)
TP_mean, TN_mean, FP_mean, FN_mean = 0, 0, 0, 0
background_model_mean = None

for i in range(1, 1100):
    curr = cv2.imread("highway/input/in%06d.jpg" % i)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    if iN < N:
        if iN == 0:
            background_model_mean = curr_gray.astype(np.float64)
        else:
            mask = cv2.inRange(BUF[:, :, iN], 0, 1)
            updated_model = np.where(mask == 0, np.mean(BUF, axis=2), background_model_mean)           
            background_model_mean = updated_model.astype(np.uint8)
        iN += 1

    else:
        mean_diff = cv2.absdiff(
            curr_gray.astype("int"), background_model_mean.astype("int")
        ).astype(np.uint8)
        mean_diff = binarization(mean_diff)

        ground_truth_mask = cv2.imread("highway/groundtruth/gt%06d.png" % i)
        ground_truth_mask = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)
        TP_mean, TN_mean, FP_mean, FN_mean = calculate_parameters(
            mean_diff, ground_truth_mask, TP_mean, TN_mean, FP_mean, FN_mean
        )

        cv2.namedWindow("mean", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mean", 1000, 1000)
        cv2.imshow("mean", mean_diff)
        cv2.waitKey(10)

    prev = curr

precision, recall, f1_score = calculate_metrics(TP_mean, TN_mean, FP_mean, FN_mean)
print("For mean")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
