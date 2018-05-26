from matplotlib import pyplot as plt
import os
import numpy as np
import cv2

print 'Reading image. . .' 

print 'Name of folder where the photos are stored: '
folder = raw_input()
#path = '.'

num_files = len([f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))])

print 'Detected ', num_files, ' files'

os.chdir(folder)
for fn in os.listdir('.'):
     if os.path.isfile(fn):
        color_img = cv2.imread(fn)
        img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('image to convert', img_gray)
        #cv2.waitKey(0)

        cv2.imwrite(fn,img_gray)


print 'Image successfully converted to grayscale!'
