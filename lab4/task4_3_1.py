import cv2
import numpy as np
from task4_1 import vis_flow

I = cv2.imread("/home/nastia/agh/avs/lab4/lab_files/I.jpg")
J = cv2.imread("/home/nastia/agh/avs/lab4/lab_files/J.jpg")
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

farnerback = cv2.calcOpticalFlowFarneback(I, J, None, 0.5, 3, 15, 3, 5, 1.2, 0)
vis_flow(farnerback[..., 0], farnerback[..., 1], I.shape, "Optical Flow Farneback")


