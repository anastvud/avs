import cv2

mandrill = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")
cv2.imshow("Mandrill", mandrill)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("/home/nastia/agh/avs/lab1/t1.png", mandrill)

print(mandrill.shape)
print(mandrill.size)
print(mandrill.dtype)
