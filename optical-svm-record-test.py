from matplotlib import pyplot as plt
#from sklearn.naive_bayes import GaussianNB
import numpy as np
import math
import cv2
import os
import os.path
from time import time

# Libraries to preform machine learning
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA

from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import accuracy_score
from sklearn import cross_validation

# Cannot use due to memory error
def pca_calc(main):
    n_components = 90000
    print '----------------------'
    print main.shape
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(main)
    main = pca.transform(main)
    print main.shape
    return main

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

def catch_first_frame():
    ret, frame_f = capf.read()
    prev_gray = cv2.cvtColor(frame_f,cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (0,0), fx=0.5, fy=0.5)
    face = face_classifier.detectMultiScale(prev_gray, 1.2, 4)

    if len(face) == 0:
        print 'No face was detected'
    else:
        print 'Face detected'
        for (x,y,w,h) in face:
            prev_gray = prev_gray[y: y+h, x: x+w]

    capf.release()
    return (x,y,w,h, prev_gray)

slash = '/'

folder_trans = np.array([])
target = np.array([])
label_trans = np.array([])
folder = ''
choice = ''

face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

print 'Load datasets [l] from file or create a new one [n]'
loading = raw_input()

if loading=='l':
    # load dataset matrix from npy file
    
    t0 = time()
    t1 = time()
    print 'Loading the main matrix. . .'
    main = np.load('datasets/optical-main.npy')
    diff = diff = time() - t1
    print 'Loaded main matrix in ', diff, 's of size ', main.shape
    
    t2 = time()

    print 'Loading the target vector. . .'
    target = np.load('datasets/optical-target.npy')
    diff = time() - t2
    print 'Loaded target in ', diff, 's of size ', target.shape

    # Getting coordinates of haar box
    print 'Getting the x, y, w, h values for Haar. . .'
    
    folder = 'datasets/Smile-short'
    first_vid = 'anthony-6-10-15_0.avi'
    capf = cv2.VideoCapture(folder + slash + first_vid)
    print 'Capf: ',capf.isOpened()
    x, y, w, h, frame_f = catch_first_frame()
else:
    while True:
        print 'Default [d] or custom [c] training set?'
        choice = raw_input()

        if choice == 'd':
            # Insert some stuff
            print 'Emotion [e] or Nodding [n] dataset?'
            type_of = raw_input()

            if type_of == 'e':
                folder_trans = np.array(['datasets/Smile-short','datasets/Shocked-short','datasets/Angry-short'])
                label_trans = np.array(['Smiling','Shocked','Angry'])
                first_vid = 'anthony-6-10-15_0.avi'
            else:
                folder_trans = np.array(['datasets/Nodding','datasets/Shaking'])
                label_trans = np.array(['Yes (nodding head)','No (shaking head)'])
                first_vid = 'anthony-6-8-15_0.avi'
            #folder_trans = np.array(['datasets/Smiling-Motion','datasets/Surprised-Motion'])
            #label_trans = np.array(['Smiling','Shocked'])
            
            
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
    face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # First frame of first video only
    if choice == 'd':
        folder = folder_trans[0]
        capf = cv2.VideoCapture(folder_trans[0] + slash + first_vid)
    else:
        capf = cv2.VideoCapture(folder + slash + first_vid)
    print 'Capf: ',capf.isOpened()
    x, y, w, h, prev_gray = catch_first_frame()

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
                frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                frame = frame[y: y+h, x: x+w]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,0.5,20,3,1,3,5,1)
                
                # Working with the flow matrix
                flow_mat = flow.flatten()  
                if j == 1:
                    sub_main = flow_mat
                elif j != 0:
                    sub_main = np.concatenate((sub_main, flow_mat))
                prev_gray = gray
                # To show us visually each video
                cv2.imshow('Optical flow',draw_flow(gray,flow))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                j = j + 1
            j = 0
                    
            if i == 0 and k == 0:
                rising_main = sub_main[None,:]
            else:
                rising_main = np.concatenate((rising_main, sub_main[None,:]))
            i = i + 1
            print rising_main.shape
        cv2.destroyAllWindows()
        i = 0
        print 'Matrix formed for ', folder_trans[k]
        k = k + 1
    main = rising_main

    np.save('optical-main.npy', main)
    np.save('optical-target.npy', target)
print 'Finished'
total_time = time() - t0
print total_time, 's'





print 'Successfully formed numpy matrix!'
print 'Now training. . .'


features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(main, target, test_size = 0.4)

# Determine amount of time to train
t0 = time()
model = SVC()
#model = SVC(kernel='poly')
#model = GaussianNB()

model.fit(features_train, labels_train)

print 'training time: ', round(time()-t0, 3), 's'

# Determine amount of time to predict
t1 = time()
pred = model.predict(features_test)

print 'predicting time: ', round(time()-t1, 3), 's'

accuracy = accuracy_score(labels_test, pred)

# Accuracy in the 0.9333, 9.6667, 1.0 range
print accuracy


# ---------------------------------
while True:
    # Test with another video
    '''
    while True:
        print "Type in the image's folder"
        folder = raw_input()

        print 'Type in the name of the image'
        fn = raw_input()

        if os.path.isfile(folder + slash + fn):
            break
        print 'File was not found, please try again'
    '''
    
    # Get user into photo position
    # Cap2 to view positon video
    cap2 = cv2.VideoCapture(0)
    
    # Manually setting x, y, w, h values in order make more consistant test
    # and training videos
    '''
    x = 103
    y = 45
    w = 163
    h = 163
    '''
    x = 90
    y = 31
    w = 181
    h = 181
    print 'Position your face until you can see it in the video box'
    print 'After that, press [q] to continue'
    while True:
        ret_f, frame_f = cap2.read()
        frame_f = cv2.resize(frame_f, (0,0), fx=0.5, fy=0.5)
        frame_f = frame_f[y: y+h, x: x+w]
        cv2.imshow('frame',frame_f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print 'You have quit'
            break
    cap2.release()

    

    # Start video to record the user
    #cap to record user for 15 frames
    cap = cv2.VideoCapture(0)

    # Name of the video file
    path = 'test.avi'

    # Starting video
    
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter(path,fourcc, 20.0, (640,480))

    print 'Press any key to start recording'
    go = raw_input()
    
    max_frames = 16
    i = 0
    # get each frame per video
    while True:
        ret, frame = cap.read()

        # Saves frame as full size
        out.write(frame)
        frame = frame[y: y+h, x: x+w]
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print 'You have quit'
            break
        i = i + 1
        # End of single sample video, save the video and move to next
        if i > max_frames:
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
        
    # To get a
    # Cap3
    cap3 = cv2.VideoCapture(path)
    ret, prev_gray = cap3.read()
    prev_gray = cv2.cvtColor(prev_gray,cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (0,0), fx=0.5, fy=0.5)
    prev_gray = prev_gray[y: y+h, x: x+w]
    
 
    face = face_classifier.detectMultiScale(prev_gray, 1.2, 4)
    '''
    if len(face) == 0:
        print 'No face was detected'
    else:
        print 'Face detected'
        for (x,y,w,h) in face:
            prev_gray = prev_gray[y: y+h, x: x+w]

    cap3.release()
    '''
    
    j = 0
    # Cap4
    cap4 = cv2.VideoCapture(path)
    while(cap4.isOpened()):
        ret, frame = cap4.read()
        if frame == None:
            cap4.release()
            break
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        frame = frame[y: y+h, x: x+w]
        cv2.imshow('To test with', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,0.5,20,3,1,3,5,1)

        # Working with the flow matrix
        flow_mat = flow.flatten()
        if j == 1:
            sub_main = flow_mat
        elif j != 0:
            sub_main = np.concatenate((sub_main, flow_mat))
        prev_gray = gray
        # To show us visually each video
        #cv2.imshow('Optical flow',draw_flow(gray,flow))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        j = j + 1


    model.predict(sub_main)

    # Determine amount of time to predict
    t1 = time()
    pred = model.predict(sub_main)

    print 'predicting time: ', round(time()-t1, 3), 's'

    print ''
    print 'Prediction: '
    print pred
