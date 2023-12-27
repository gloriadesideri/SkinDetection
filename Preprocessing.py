import cv2 as cv
import numpy as np
import imutils
import matplotlib
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageEnhance

class Preprocessing:
    
    def __init__(self, datapath):
        self.datapath = datapath
        self.__images=None
        self.__uplighting=None
        self.__downlighting=None
        self.__mask=None
        self.__load_images()
        self.__load_mask()
        self.__data_augmentation()
    
    def __load_images(self):
        """
        Load all the images from datapath. 
        """
        if not os.path.exists(os.path.join(self.datapath, 'images')):
            print(f"The specified folder path '{self.datapath}' does not exist.")
        else:
            # Get a list of all files in the specified folder
            files = os.listdir(os.path.join(self.datapath, 'images'))
            image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]  

            # List to store the loaded images
            loaded_images = []

            # Loop through each image file and load it using OpenCV
            for image_file in image_files:
                image_folder = os.path.join(self.datapath, 'images')
                image_path = os.path.join(image_folder, image_file)
                img = cv.imread(image_path)

                if img is not None:
                    # Append the loaded image to the list
                    loaded_images.append(img)
            self.__images=loaded_images
            
    def __load_mask(self):
        """
        Load all the masks from datapath. 
        """
        if not os.path.exists(os.path.join(self.datapath, 'masks')):
            print(f"The specified folder path '{self.datapath}' does not exist.")
        else:
            # Get a list of all files in the specified folder
            files = os.listdir(os.path.join(self.datapath, 'masks'))
            image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]  

            # List to store the loaded images
            loaded_images = []

            # Loop through each image file and load it using OpenCV
            for image_file in image_files:
                image_folder = os.path.join(self.datapath, 'masks')
                image_path = os.path.join(image_folder, image_file)
                img = cv.imread(image_path)

                if img is not None:
                    # Append the loaded image to the list
                    loaded_images.append(img)
            self.__mask=loaded_images
            
            
    def __data_augmentation(self):
        """
        Change the lightling of the images
        """
        # Convert OpenCV images to PIL images
        pil_images = []
        for cv_image in self.__images:
            # Convert BGR to RGB as PIL works with RGB
            cv_image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image_rgb)
            pil_images.append(pil_image)
        # Enhance the brightness to generate versions with different lighting conditions
        generated_images_cv2_up=[]
        generated_images_cv2_down=[]
        for pil_image in pil_images:
            # Create an ImageEnhance object
            enhancer = ImageEnhance.Brightness(pil_image)

            # Generate the version with poor lighting (reduced brightness)
            poor_brightness = enhancer.enhance(0.5)
            poor_brightness_cv2 = cv.cvtColor(np.array(poor_brightness), cv.COLOR_RGB2BGR)
            generated_images_cv2_up.append(poor_brightness_cv2)

            # Generate the version with too bright lighting
            too_bright = enhancer.enhance(1.5)
            too_bright_cv2 = cv.cvtColor(np.array(too_bright), cv.COLOR_RGB2BGR)
            generated_images_cv2_down.append(too_bright_cv2)
        self.__uplighting=generated_images_cv2_up
        self.__downlighting=generated_images_cv2_down
        
        pass
    def get_images(self):
        """
        Return the images
        """
        return self.__images, self.__uplighting, self.__downlighting, self.__mask