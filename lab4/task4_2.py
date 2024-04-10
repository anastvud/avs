import cv2
import numpy as np
from task4_1 import of, vis_flow


def pyramid(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k - 1], (0, 0), fx=0.5, fy=0.5))
    return images

def optical_flow_multiscale(I_org, J, max_scale=3, W2=3, dY=3, dX=3):
    IP = pyramid(I_org, max_scale)
    I = IP[-1] 

    for scale in range(max_scale - 1, -1, -1):
        u_scale, v_scale = of(IP[scale], I, J, W2, dY, dX)
        
        if scale > 0:
            I_new = cv2.resize(I, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            for j in range(I_new.shape[0]):
                for i in range(I_new.shape[1]):
                    if 0 <= j - v_scale[j, i] * 2 < I.shape[0] and 0 <= i - u_scale[j, i] * 2 < I.shape[1]:
                        I_new[j, i] = I[j - v_scale[j, i] * 2, i - u_scale[j, i] * 2]
            I = I_new

    u_total = np.zeros_like(u_scale)
    v_total = np.zeros_like(v_scale)

    for scale in range(max_scale):
        u_scale, v_scale = of(IP[scale], I_org, J, W2, dY, dX)
        u_total += u_scale
        v_total += v_scale

    vis_flow(u_total, v_total, I_org.shape, "Optical Flow Multiscale")

I = cv2.imread("/home/nastia/agh/avs/lab4/lab_files/I.jpg")
J = cv2.imread("/home/nastia/agh/avs/lab4/lab_files/J.jpg")
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

optical_flow_multiscale(I, J)