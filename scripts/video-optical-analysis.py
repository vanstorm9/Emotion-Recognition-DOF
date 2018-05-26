from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import os
from time import time

def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis

slash = '/'

folder_trans = np.array([])
target = np.array([])
label_trans = np.array([])
folder = ''
choice = ''

while True:
    print 'Default [d] or custom [c] training set?'
    choice = raw_input()

    if choice == 'd':
        # Insert some stuff
        folder_trans = np.array(['../datasets/Smiling-Motion','datasets/Surprised-Motion'])
        label_trans = np.array(['Smiling','Shocked'])
        first_vid = 'anthony-6-6-15_0.avi'
        break
    elif choice == 'c':
        print 'Type in number of folders:'
        iterations = raw_input()
        i = 0
        while i < int(iterations):
            print 'Type in folder name:'
            folder = raw_input()

            trans_fold_array = np.array([folder])
            folder_trans = np.concatenate((folder_trans,trans_fold_array))
            
            print 'Which video would you like the first frame to be from?'
            first_vid = raw_input()
            i = i + 1

            print 'Name of Label'
            label = raw_input()

            trans_array = np.array([label])
            label_trans = np.concatenate((label_trans,trans_array))
        break

# Detect the first frame of video
face_classifier = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

# First frame of first video only
if choice == 'd':
    folder = folder_trans[0]
    capf = cv2.VideoCapture(folder_trans[0] + slash + first_vid)
else:
    capf = cv2.VideoCapture(folder + slash + first_vid)
print 'Capf: ',capf.isOpened()
ret, frame_f = capf.read()
prev_gray = cv2.cvtColor(frame_f,cv2.COLOR_BGR2GRAY)
face = face_classifier.detectMultiScale(prev_gray, 1.2, 4)

if len(face) == 0:
    print 'No face was detected'
else:
    print 'Face detected'
    for (x,y,w,h) in face:
        prev_gray = prev_gray[y: y+h, x: x+w]

capf.release()

num_files = len([f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))])
print num_files
k = 0
i = 0
j = 0
p = 0
main = np.array([[[[[]]]]])
rising_main = np.array([[[[[]]]]])


t0 = time()
# This loop is to go through all the chosen folders
while k < label_trans.size:
    # Form the target label as we go through each video
    num_files = len([f for f in os.listdir(folder_trans[k])
                if os.path.isfile(os.path.join(folder_trans[k], f))])
    p = 0
    while p < num_files:
        label_array = np.array([label_trans[k]])
        if p == 0 and k==0:
            target = label_array
        else:
            target = np.concatenate((target,label_array))
        label_array = np.array([])
        p = p + 1

    
    # The main loop for going through all videos within a single chosen folder
    for fn in os.listdir(folder_trans[k]):
        cap = cv2.VideoCapture(folder_trans[k] + slash + fn)
        # This loops is to go through each frame per video
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame == None:
                cap.release()
                break
            frame = frame[y: y+h, x: x+w]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,0.5,1,3,1,3,5,1)

            # Working with the flow matrix
            flow_mat = flow.flatten()
            if j == 0:
                sub_main = flow_mat[None,:]
            else:
                sub_main = np.concatenate((sub_main, flow_mat[None,:]))
            prev_gray = gray
            # To show us visually each video
            #cv2.imshow('Optical flow',draw_flow(gray,flow))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            j = j + 1
        j = 0
        

        if i == 0:
            rising_main = sub_main[None,:]
        else:
            rising_main = np.concatenate((rising_main, sub_main[None,:]))
        
        i = i + 1
    cv2.destroyAllWindows()
    i = 0
    print 'Matrix formed for ', folder_trans[k]

    if k == 0:
        main = rising_main[None,:]
    else:
        main = np.concatenate((main,rising_main[None,:]))
    k = k + 1
print 'Finished'
total_time = time() - t0
print total_time, 's'
