import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2


def grdient(channel):
    dx = scipy.ndimage.filters.convolve1d(channel.astype(np.int32), np.array([-1, 0, 1]), axis=1)
    dy = scipy.ndimage.filters.convolve1d(channel.astype(np.int32), np.array([-1, 0, 1]), axis=0)
    
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * (180 / np.pi) % 180  # Convert radians to degrees and normalize to [0, 180)
    # print(orientation)
    
    return magnitude, orientation


def max_mag_orient(image):
    if len(image.shape) == 2:  # Grayscale image   
        magnitude, orientation = grdient(image)
    
    elif len(image.shape) == 3:  # RGB image
        magnitudes = []
        orientations = []

        for i in range(3):
            mag, orient = grdient(image[:, :, i])
            magnitudes.append(mag)
            orientations.append(orient)
        
        magnitudes = np.stack(magnitudes, axis=-1)
        orientations = np.stack(orientations, axis=-1)
        
        magnitude = np.max(magnitudes, axis=-1)
        orientation = np.zeros_like(magnitude)
        
        for i in range(3):
            orientation = np.where(magnitudes[:, :, i] == magnitude, orientations[:, :, i], orientation)
        
        return magnitude, orientation



image_dir = os.getenv("REPO_ROOT") + "/lab11/"

img = cv2.imread(image_dir + "test_images/testImage4.png")
magn, orient = max_mag_orient(img)

