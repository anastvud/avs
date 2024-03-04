import cv2
import numpy as np
import matplotlib.pyplot as plt


madrill = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")
lena = cv2.imread("/home/nastia/agh/avs/lab1/lena.png")

m_gray = cv2.cvtColor(madrill, cv2.COLOR_BGR2GRAY)
l_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

# resize m_gray to add two pictures
m_gray = cv2.resize(m_gray, (l_gray.shape[1], l_gray.shape[0]))


def hist(img):
    h = np.zeros((256, 1), np.float32)  # creates and zeros single - column arrays
    height, width = img.shape[:2]  # shape - we take the first 2 values
    for y in range(height):
        for x in range(width):
            intensity = img[y, x]
            h[intensity] += 1
    return h


hist_func = cv2.calcHist([m_gray], [0], None, [256], [0, 256])
hist_cust = hist(m_gray)

fig, agx = plt.subplots(2, 1, figsize=(8, 6))

agx[0].plot(hist_func, color="red")
agx[1].plot(hist_cust, color="blue")
plt.show()
