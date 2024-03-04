import cv2
import numpy as np
import matplotlib.pyplot as plt


madrill = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")
lena = cv2.imread("/home/nastia/agh/avs/lab1/lena.png")

m_gray = cv2.cvtColor(madrill, cv2.COLOR_BGR2GRAY)
l_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

# resize m_gray to add two pictures
m_gray = cv2.resize(m_gray, (l_gray.shape[1], l_gray.shape[0]))

sum_gray = np.add(m_gray, l_gray)
diff_gray = np.subtract(m_gray, l_gray)
mult_gray = np.multiply(m_gray, l_gray)

# linear combination
alpha = 0.5
beta = 1.0 - alpha
lin_comb = cv2.addWeighted(m_gray, alpha, l_gray, beta, 0.0)

# absdiff
mod_diff = cv2.absdiff(m_gray, l_gray)


plt.figure(figsize=(18, 10))

plt.subplot(2, 4, 1)
plt.title("m_gray")
plt.imshow(cv2.cvtColor(m_gray, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 2)
plt.title("l_gray")
plt.imshow(cv2.cvtColor(l_gray, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 3)
plt.title("sum_gray")
plt.imshow(cv2.cvtColor(sum_gray, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 4)
plt.title("diff_gray")
plt.imshow(cv2.cvtColor(diff_gray, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 5)
plt.title("mult_gray")
plt.imshow(cv2.cvtColor(mult_gray, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 6)
plt.title("lin_comb")
plt.imshow(cv2.cvtColor(lin_comb, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 7)
plt.title("mod_diff")
plt.imshow(cv2.cvtColor(mod_diff, cv2.COLOR_BGR2RGB))

plt.show()
