import cv2
import numpy as np
import matplotlib.pyplot as plt


madrill = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")
lena = cv2.imread("/home/nastia/agh/avs/lab1/lena.png")

m_gray = cv2.cvtColor(madrill, cv2.COLOR_BGR2GRAY)
l_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
m_gray = cv2.resize(m_gray, (l_gray.shape[1], l_gray.shape[0]))

plt.figure(figsize=(18, 10))

plt.subplot(2, 4, 1)
plt.title("m_gray")
plt.imshow(cv2.cvtColor(m_gray, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 2)
plt.title("l_gray")
plt.imshow(cv2.cvtColor(l_gray, cv2.COLOR_BGR2RGB))

# change to int32 to add two pictures
m_gray = m_gray.astype('int')
l_gray = l_gray.astype('int')

sum_gray = np.add(m_gray, l_gray)
diff_gray = np.subtract(m_gray, l_gray)
mult_gray = np.multiply(m_gray, l_gray)

# linear combination
alpha = 0.5
beta = 1.0 - alpha
lin_comb = cv2.addWeighted(m_gray, alpha, l_gray, beta, 0.0)

# absdiff
mod_diff = cv2.absdiff(m_gray, l_gray)

info = np.iinfo(m_gray.dtype)
m_gray = ((m_gray / info.max) * 255).astype(np.uint8)
l_gray = ((l_gray / info.max) * 255).astype(np.uint8)

plt.subplot(2, 4, 3)
plt.title("sum_gray")
plt.imshow(sum_gray, cmap='gray')

plt.subplot(2, 4, 4)
plt.title("diff_gray")
plt.imshow(diff_gray, cmap='gray')

plt.subplot(2, 4, 5)
plt.title("mult_gray")
plt.imshow(mult_gray, cmap='gray')

plt.subplot(2, 4, 6)
plt.title("lin_comb")
plt.imshow(lin_comb, cmap='gray')

plt.subplot(2, 4, 7)
plt.title("mod_diff")
plt.imshow(mod_diff, cmap='gray')

plt.show()
