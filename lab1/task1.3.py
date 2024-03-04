import cv2

i = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")

IG = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

cv2.imshow("IG", IG)
cv2.imshow("IHSV", IHSV)
cv2.waitKey(0)
cv2.destroyAllWindows()
