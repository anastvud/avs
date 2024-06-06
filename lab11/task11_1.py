import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2

def compute_gradients(image):
    if len(image.shape) == 2:  # Grayscale image
        dx = scipy.ndimage.filters.convolve1d(image.astype(np.int32), np.array([-1, 0, 1]), axis=1)
        dy = scipy.ndimage.filters.convolve1d(image.astype(np.int32), np.array([-1, 0, 1]), axis=0)
        
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * (180 / np.pi) % 180  # Convert radians to degrees and normalize to [0, 180)
        
        return magnitude, orientation
    
    elif len(image.shape) == 3:  # RGB image
        magnitudes = []
        orientations = []
        
        # Compute gradients for each channel
        for i in range(3):
            dx = scipy.ndimage.filters.convolve1d(image.astype(np.int32), np.array([-1, 0, 1]), axis=1)
            dy = scipy.ndimage.filters.convolve1d(image.astype(np.int32), np.array([-1, 0, 1]), axis=0)
            
            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = np.arctan2(dy, dx) * (180 / np.pi) % 180  # Convert radians to degrees and normalize to [0, 180)
            magnitudes.append(magnitude)
            orientations.append(orientation)
        
        return magnitudes, orientations


max_B = np . logical_and ( magnitude [: , :, 1] < magnitude [: , : , 0] , magnitude[: , : , 2] < magnitude [: , :, 0])