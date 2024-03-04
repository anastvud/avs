import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


i = plt.imread("/home/nastia/agh/avs/lab1/mandrill.jpg")
plt.figure(1)
plt.imshow(i)
plt.title("Mandrill")
plt.axis("off")
plt.show()
plt.imsave("lab1/t2.png", i)

x = [100, 150, 200, 250]
y = [50, 100, 150, 200]
plt.plot(x, y, "r.", markersize=10)
plt.show()

fig, ax = plt.subplots(1)
rect = Rectangle((50, 50), 50, 00, fill=False, ec="r")
ax.add_patch(rect)
plt.show()
