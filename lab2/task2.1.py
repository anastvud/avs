import cv2
import numpy as np


prev = cv2.imread('pedestrian/input/in000300.jpg')
TP, TN, FP, FN = 0, 0, 0, 0

for i in range (301, 1100) :

    curr = cv2.imread('pedestrian/input/in%06d.jpg' % i)
    
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # _, curr_gray = cv2.threshold(curr_gray, 145, 255, cv2.THRESH_BINARY, )
    # _, prev_gray = cv2.threshold(prev_gray, 145, 255, cv2.THRESH_BINARY, )    
    
    curr_gray = curr_gray.astype('int')
    prev_gray = prev_gray.astype('int')

    
    mod_diff = cv2.absdiff(prev_gray, curr_gray)


    mod_diff = mod_diff.astype(np.uint8)


    # medianBlur
    mod_diff = cv2.medianBlur(mod_diff, 5)

    # # # dilating and eroding
    # kernel = np.ones((5, 5), np.uint8) 
    # mod_diff = cv2.dilate(mod_diff, kernel, iterations=1) 
    # mod_diff = cv2.erode(mod_diff, kernel, iterations=1) 



    # # labeling
    # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mod_diff)
    # cv2.imshow("Labels", np.uint8(labels / retval * 255))

    # I_VIS = mod_diff # copy of the input image
    # if (stats.shape[0] > 1) : # are there any objects
    #     tab = stats [1:, 4] # 4 columns without first element
    #     pi = np.argmax(tab)# finding the index of the largest item
    #     pi = pi + 1 # increment because we want the index in stats , not in tab
    #     # drawing a bbox
    #     cv2.rectangle(I_VIS, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]), (255, 0, 0), 2)
    #     # print information about the field and the number of the largest element
    #     cv2.putText(I_VIS,"%f" % stats[pi ,4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX ,0.5 ,(255 ,0 ,0))
    #     cv2.putText(I_VIS ,"%d" %pi, (np.int32(centroids[pi, 0]), np.int32(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))





    cv2.namedWindow("Resized_Window1", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Resized_Window1", 1000, 1000) 
    cv2.imshow("Resized_Window1", mod_diff)
    cv2.waitKey(10)



    # cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
    # cv2.resizeWindow("Resized_Window", 1000, 1000) 
    # cv2.imshow("Resized_Window", prev_gray)
    # cv2.waitKey(10)

    prev = curr