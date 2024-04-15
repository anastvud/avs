import cv2
import numpy as np
from task4_1 import vis_flow

I = cv2.imread("/home/nastia/agh/avs/lab4/lab_files/I.jpg")
J = cv2.imread("/home/nastia/agh/avs/lab4/lab_files/J.jpg")
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

farnerback = cv2.calcOpticalFlowFarneback(I, J, None, 0.5, 3, 15, 3, 5, 1.2, 0)
vis_flow(farnerback[..., 0], farnerback[..., 1], I.shape, "Optical Flow Farneback")


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
 qualityLevel = 0.3,
 minDistance = 7,
 blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
 maxLevel = 2,
 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it                  
p0 = cv2.goodFeaturesToTrack(I, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(I)
p1, st, err = cv2.calcOpticalFlowPyrLK(I, J, p0, None, **lk_params)
# Select good points
if p1 is not None:
    good_new = p1[st==1]
    good_old = p0[st==1]
# draw the tracks
img = np.copy(J)
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    J = cv2.circle(J, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(J, mask)

cv2.namedWindow("name", cv2.WINDOW_NORMAL)
cv2.resizeWindow("name", 1000, 1000)
cv2.imshow("name", img)
cv2.waitKey(0)
cv2.destroyAllWindows()