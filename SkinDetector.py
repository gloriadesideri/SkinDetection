"""
    1. Load the image
    2. Data augmentation to change the lightling
    3. Detect the skin area
    4. Compare with true values
"""
import cv2 as cv
import numpy as np
import imutils
import matplotlib
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageEnhance
from Preprocessing import Preprocessing


DATAPATH= 'hands/'

if __name__ == "__main__":
    preprocessing = Preprocessing(DATAPATH)
    images , uplighting, downlighting, mask = preprocessing.get_images()
    print(len(images))
    print(len(uplighting))
    print(len(downlighting))
    print(len(mask))

        
    
    