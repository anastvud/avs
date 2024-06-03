import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


image_dir = os.getenv("REPO_ROOT") + "/lab10/"


cap = cv2.VideoCapture(image_dir + 'vid1_IR.avi')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




    _, img_bin = cv2.threshold(img_gray, 35, 255, cv2.THRESH_BINARY)
    img_bin = cv2.medianBlur(img_bin, 5)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area < 1500 or h < w:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('Frame',frame)







    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
  else: 
    break
    
