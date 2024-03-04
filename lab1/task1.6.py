import cv2
import matplotlib.pyplot as plt

madrill = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")

m_gray = cv2.cvtColor(madrill, cv2.COLOR_BGR2GRAY)
IGE = cv2.equalizeHist(m_gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
I_CLAHE = clahe.apply(m_gray)


plt.figure(figsize=(18, 10))

plt.subplot(1, 3, 1)
plt.title("m_gray")
plt.imshow(cv2.cvtColor(m_gray, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("IGE")
plt.imshow(cv2.cvtColor(IGE, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("I_CLAHE")
plt.imshow(cv2.cvtColor(I_CLAHE, cv2.COLOR_BGR2RGB))

plt.show()
