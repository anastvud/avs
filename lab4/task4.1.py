import cv2
import numpy as np

W2 = 3
dX = dY = 3

I = cv2.imread('/home/nastia/agh/avs/lab4/lab_files/I.jpg')
J = cv2.imread('/home/nastia/agh/avs/lab4/lab_files/J.jpg')

I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

u = np.zeros((I.shape[0], I.shape[1]), dtype=np.float32)
v = np.zeros((I.shape[0], I.shape[1]), dtype=np.float32)

mod_diff = cv2.absdiff(I.astype("int"), J.astype("int")).astype(np.uint8)
    
for j in range(W2, I.shape[0] - W2):
    for i in range(W2, I.shape[1] - W2):
        IO = np.float32(I[j - W2:j + W2 + 1, i - W2:i + W2 + 1])

        min_distance = float('inf')
        min_x, min_y = 0, 0

        for dj in range(-dY, dY + 1):
            for di in range(-dX, dX + 1):
                if j + dj >= W2 and j + dj < I.shape[0] - W2 and i + di >= W2 and i + di < I.shape[1] - W2:
                    JO = np.float32(J[j + dj - W2:j + dj + W2 + 1, i + di - W2:i + di + W2 + 1])
                    distance = np.sqrt(np.sum(np.square(JO - IO)))
                    if distance < min_distance:
                        min_distance = distance
                        min_x, min_y = di, dj

        u[j, i] = min_x
        v[j, i] = min_y            


magnitude, angle = cv2.cartToPolar(u, v)
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
angle = (angle * 90 / np.pi).astype(np.uint8)

optical_flow_hsv = np.zeros((I.shape[0], I.shape[1], 3), dtype=np.uint8)
optical_flow_hsv[..., 0] = angle
optical_flow_hsv[..., 1] = 255
optical_flow_hsv[..., 2] = magnitude
optical_flow_bgr = cv2.cvtColor(optical_flow_hsv, cv2.COLOR_HSV2BGR)

cv2.namedWindow("mod_diff", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mod_diff", 1000, 1000)
cv2.imshow("mod_diff", optical_flow_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
