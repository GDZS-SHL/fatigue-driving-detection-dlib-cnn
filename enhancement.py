# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:33:11 2021

@author: 34296
"""

import cv2
import numpy as np
def enhancement(img,c=0.4,scale=256):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    sobel=np.absolute(sobely)+np.absolute(sobelx)
    sobel=IntensityScratch(sobel)
    sobel_smoothed=cv2.GaussianBlur(sobel,(5,5),0)
    lpc=cv2.Laplacian(img,cv2.CV_64F)
    lpc=IntensityScratch(lpc)
    product=IntensityScratch(sobel_smoothed*lpc).astype(np.uint8)
    img_out=IntensityScratch(img+c*product).astype(np.uint8)
    return img_out
def gamma(img, gamma=0.6,scale=256):
    table = np.array([((i / (scale-1)) ** gamma) * (scale-1) for i in np.arange(0, scale)]).astype("uint8")
    return cv2.LUT(img, table)
def IntensityScratch(img,scale=256):
    return np.around((img-img.min())/(img.max()-img.min())*(scale-1))

if __name__ == '__main__':
    img=cv2.imread('C:/Users/34296/Desktop/eye/poorimg.png',cv2.IMREAD_GRAYSCALE)
    img_out=gamma(enhancement(img),0.1)
    cv2.imshow('img',img)
    cv2.imshow('img_out',img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()