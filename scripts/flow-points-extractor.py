import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def form_image_trait(folder):
     main = np.array([[[]]])
     z = 0
     #os.chdir(folder)
     for fn in os.listdir(folder):
          if os.path.isfile(fn):
             img = mpimg.imread(fn)
             if z == 0:
                  main = np.array([img])
                  z = z + 1
                  continue
             img_mat = np.array([img])
             main = np.concatenate((main,img_mat), axis=0)
             z = z + 1
     return main

# This is working
def form_data_trait(folder):
     main = np.array([[]])
     z = 0
     for fn in os.listdir(folder):
          slash = '/'
          img = mpimg.imread(folder + slash + fn)
          img = img.flatten()
          if z == 0:
               main = np.array(img)
               z = z + 1
               continue
          img_mat = np.array(img)
          main = np.vstack((main,img_mat))
          z = z + 1
     return main

target = np.array([])
label_trans = np.array([])
folder_trans = np.array([])

print 'Reading image. . .' 

print 'Number of folders: '
iterations = raw_input()
iterations = int(iterations)
k = 0
while k < iterations:   
     print 'Name of folder where the photos are stored: '
     folder = raw_input()

     trans_fold_array = np.array([folder])
     folder_trans = np.concatenate((folder_trans,trans_fold_array))
     
     print 'Name of Label'
     label = raw_input()
     
     trans_array = np.array([label])
     label_trans = np.concatenate((label_trans,trans_array))
     k = k + 1
#path = '.'


i = 0
j = 0

main = np.array([[]])
while j < label_trans.size:
     num_files = len([f for f in os.listdir(folder_trans[j])
                if os.path.isfile(os.path.join(folder_trans[j], f))])

     print 'Detected ', num_files, ' files'

     # While loop creates the target/label array
     while i < num_files:
          label_array = np.array([label_trans[j]])
          target = np.concatenate((target,label_array))
          i = i + 1
     add_to_main = form_data_trait(folder_trans[j])

     # intitalize inital matrix, else concatenate to previous main matrix
     if j == 0:
          main = add_to_main
     else:
          main = np.concatenate((main, add_to_main))
     j = j + 1
     i = 0


print 'Successfully formed numpy matrix!'
