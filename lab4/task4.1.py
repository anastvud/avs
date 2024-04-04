import cv2
import numpy as np

I = cv2.imread('/home/nastia/agh/avs/lab4/lab_files/I.jpg')
J = cv2.imread('/home/nastia/agh/avs/lab4/lab_files/J.jpg')

I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

mod_diff = cv2.absdiff(I.astype("int"), J.astype("int")).astype(np.uint8)
    

cv2.namedWindow("mod_diff", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mod_diff", 1000, 1000)
cv2.imshow("mod_diff", mod_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
