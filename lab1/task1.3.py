import cv2

i = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")

IG = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

H, S, V = IHSV[:,:,0], IHSV[:,:,1], IHSV[:,:,2]

cv2.imshow("IG", IG)
cv2.imshow("IHSV", IHSV)
cv2.imshow("H", H)
cv2.imshow("S", S)
cv2.imshow("V", V)
cv2.waitKey(0)
cv2.destroyAllWindows()
