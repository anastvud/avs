import cv2

i = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")
height, width = i.shape[:2]
scale = 1.75
Ix2 = cv2.resize(i, (int(scale * height), int(scale * width)))
cv2.imshow("Ix2", Ix2)
cv2.waitKey(0)
cv2.destroyAllWindows()
