from matplotlib import pyplot as plt
from PIL import Image
import os
import numpy as np
import cv2


def find_smallest(folder):
    smallest = float("inf")
    name = ''
    os.chdir(folder)
    for fn in os.listdir('.'):
         if os.path.isfile(fn):
            img = Image.open(fn)
            curr_size = img.size[0]
            if smallest > curr_size:
                smallest = curr_size
                name = fn
            #print curr_size
    return smallest


print 'Name of folder where the photos are stored: '
folder = raw_input()


#path = '.'

num_files = len([f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))])


print 'Detected ', num_files, ' files'
print 'Searching for smallest length'
#length = find_smallest(folder)
length = 214
print 'Smallest length is: ', length

os.chdir(folder)
for fn in os.listdir('.'):
    if os.path.isfile(fn):
        img = Image.open(fn)
        img.resize((length,length)).save(fn)
print 'Successfully resized all images!'










