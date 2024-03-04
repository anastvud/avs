import cv2
import matplotlib.pyplot as plt

madrill = cv2.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")

# gaussian
gaussian_blur = cv2.GaussianBlur(madrill, (5, 5), cv2.BORDER_DEFAULT)

# sobel
grad_x = cv2.Sobel(
    madrill, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
)
grad_y = cv2.Sobel(
    madrill, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
s_dst = cv2.convertScaleAbs(sobel)

# laplacian
laplacian = cv2.Laplacian(madrill, cv2.CV_16S)
abs_dst = cv2.convertScaleAbs(laplacian)

# median
median = cv2.medianBlur(madrill, 5)


plt.figure(figsize=(18, 10))

plt.subplot(2, 2, 1)
plt.axis("off")
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.axis("off")
plt.imshow(cv2.cvtColor(s_dst, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 3)
plt.axis("off")
plt.imshow(cv2.cvtColor(abs_dst, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 4)
plt.axis("off")
plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))

plt.show()
