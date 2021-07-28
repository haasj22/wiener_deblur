"""
AUTHOR: John Haas
DATE CREATED: July 26, 2021

DESCRIPTION: code that deblurs a bunch of images
NOTES: 
-User needs to change filepath to location of folder
-Currently only tests on the first image
-Having an issue with overfitting as results are not that good
"""
import os
import cv2
from scipy.misc import face
from skimage import color, data, restoration
import matplotlib.pyplot as plt
import numpy as np

#variables to store best angle and best strength
best_strength = None
best_angle = None
best_matrix = None

#CHANGE THIS TO WHERE IMAGES ARE LOCATED
#place where images are stored
filepath = '/mnt/storage2/METRO_recycling/john_images_for_IIC'

#goes through images in given directory
for image in os.listdir(filepath):

    #reads the image, converts it to grey
    image = cv2.imread(filepath + "/" + image)
    gray = color.rgb2gray(image)
    
    #Calculates the Tenengrad of the image
    #SRC: rbaron.net/blog/2020/02/16/How-to-identify-blurry-images.html
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 5)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 5)
    best_blur = cv2.magnitude(sx, sy).mean()
    
    #Use this line if you want to optimize the Lapacian instead of the Tenengrad
    """
    Currently not used as we predict it is more likely to overfit
    as the wiener algorithm optimizes the Laplacian
    """
    #best_blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    print("First blur: " + str(best_blur))
    
    #shows the image that will be deblured
    plt.imshow(gray)
    plt.show()
    
    #used to store current psf_matrix we are using
    strength = None
    angle = None
    psf_matrix = None


    #goes through file containing psf_matrixes and reads it line by line
    f = open("all_psf.txt")
    for line in f:
        if line[0] == "i":

            #if the matrix is not empty
            if not (strength == None and angle == None):
                print("PSF matrix")
                print(psf_matrix)

                #deconvolutes image using given filter and measures blur again
                """
                wiener function takes a gray image and a matrix description of
                the blur=, called a PSF to deblur that given image
                """
                filtered_image = restoration.unsupervised_wiener(gray, psf_matrix.astype(float))
                print("Filtered")
                print(filtered_image)
                
                #measures new blur
                sx = cv2.Sobel(filtered_image[0], cv2.CV_64F, 1, 0, ksize = 5)
                sy = cv2.Sobel(filtered_image[0], cv2.CV_64F, 0, 1, ksize = 5)
                new_blur = cv2.magnitude(sx, sy).mean()
                
                #new_blur = cv2.Laplacian(filtered_image[0], cv2.CV_64F).var()

                #if blur is better, record it
                if new_blur >0:
                    #plt.imshow(filtered_image[0])
                    #plt.show()
                    pass
                print("New blur: " + str(new_blur))
                if new_blur > best_blur:
                    best_blur = new_blur
                    best_strength = strength
                    best_angle = angle
                    best_matrix = psf_matrix
                psf_matrix = None
                
            #read the new strength
            strength = int(line.strip().split(" ")[-1])
            print("Testing strength: " + str(strength)) 
        elif line[0] == "j":
            #read the new angle
            angle = int(line.strip().split(" ")[-1])
            print("Testing angle: " + str(angle))
        else:
            #reads the psf matrix and stores it 
            if psf_matrix is  None:
                psf_matrix = np.array([line.strip().split("\t")])
            else:
                new_array = np.array(line.strip().split("\t"))
                psf_matrix = np.vstack([psf_matrix, new_array])
    f.close()
    
    print("here")
    #print the optimized parameters
    print(best_strength)
    print(best_angle)
    print(best_matrix)
    
    #shows the original image and its optimized results
    plt.imshow(gray)
    plt.show()
    filtered_image = restoration.unsupervised_wiener(gray, best_matrix.astype(float))
    plt.imshow(filtered_image[0])
    plt.show()

    break
