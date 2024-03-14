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


with open("pedestrian/temporalROI.txt", "r") as file:
    line = file.readline()
    roi_start, roi_end = map(int, line.split())

prev = cv2.imread("pedestrian/input/in000300.jpg")
TP, TN, FP, FN = 0, 0, 0, 0

for i in range(roi_start, roi_end):

    curr = cv2.imread("pedestrian/input/in%06d.jpg" % i)

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    mod_diff = cv2.absdiff(prev_gray.astype("int"), curr_gray.astype("int")).astype(
        np.uint8
    )
    _, mod_diff = cv2.threshold(mod_diff, 17, 255, cv2.THRESH_BINARY)

    # medianBlur
    mod_diff = cv2.medianBlur(mod_diff, 5)

    # # dilating and eroding
    kernel = np.ones((5, 5), np.uint8)
    mod_diff = cv2.dilate(mod_diff, kernel, iterations=1)
    mod_diff = cv2.erode(mod_diff, kernel, iterations=1)

    # calculate parameters
    ground_truth_mask = cv2.imread("pedestrian/groundtruth/gt%06d.png" % i)
    ground_truth_mask = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)

    TP_M = np.logical_and((mod_diff == 255), (ground_truth_mask == 255))
    TP_S = np.sum(TP_M)
    TP += TP_S

    FP_M = np.logical_and((mod_diff == 255), (ground_truth_mask == 0))
    FP_S = np.sum(FP_M)
    FP += FP_S

    FN_M = np.logical_and((mod_diff == 0), (ground_truth_mask == 255))
    FN_S = np.sum(FN_M)
    FN += FN_S

    TN_M = np.logical_and((mod_diff == 0), (ground_truth_mask == 0))
    TN_S = np.sum(TN_M)
    TN += TN_S

    # labeling
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mod_diff)
    cv2.imshow("Labels", np.uint8(labels / retval * 255))

    I_VIS = curr_gray  # copy of the input image
    if stats.shape[0] > 1:  # are there any objects
        tab = stats[1:, 4]  # 4 columns without first element
        pi = np.argmax(tab)  # finding the index of the largest item
        pi = pi + 1  # increment because we want the index in stats , not in tab
        # drawing a bbox
        cv2.rectangle(
            I_VIS,
            (stats[pi, 0], stats[pi, 1]),
            (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]),
            (0, 0, 0),
            2,
        )
        # print information about the field and the number of the largest element
        cv2.putText(
            I_VIS,
            "%f" % stats[pi, 4],
            (stats[pi, 0], stats[pi, 1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )
        cv2.putText(
            I_VIS,
            "%d" % pi,
            (np.int32(centroids[pi, 0]), np.int32(centroids[pi, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
        )

    cv2.namedWindow("Resized_Window1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window1", 1000, 1000)
    cv2.imshow("Resized_Window1", I_VIS)
    cv2.waitKey(10)

    prev = curr

precision, recall, f1_score = calculate_metrics(TP, TN, FP, FN)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
