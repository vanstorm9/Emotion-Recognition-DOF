# Program implemented in the Raspberry Pi (with camera module)

#from matplotlib import pyplot as plt
#from sklearn.naive_bayes import GaussianNB
import numpy as np
import math
import cv2
import os
import os.path
import io
from time import time
import picamera

import smtplib

#camera = picamera.PiCamera()

from time import sleep

#import pyttsx

# Libraries to preform machine learning
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,accuracy_score, confusion_matrix

from sklearn.decomposition import PCA, RandomizedPCA

# from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib

from sklearn import cross_validation
from sklearn.linear_model import Ridge
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.externals import joblib


def emotion_to_text(pred):

    smtpUser= "(ENTER YOUR EMAIL ADDRESS)"
    smtpPass= "(ENTER YOUR EMAIL ACCOUNT'S PASSWORD)"

    toAdd = "19739790997@tmomail.net"
    fromAdd = smtpUser
    
    if pred == "Neutral":
        subject = "How are you doing?"
        body = "Hey! Just checking in, I was just wondering how you are doing today. \n \n  - Rapiro"
    elif pred == "Angry":
        subject = "Are you okay? You look mad"
        body = "I noticed that you are a bit red. Did something annoy or aggrivate you? /n -Rapiro"
    elif pred == "Shocked":
        subject = "Did something scare or surprised you?"
        body = "What's wrong, you look like you have seen a ghost. . . \n Rapiro"
    else:
        subject = "You seem happy today"
        body = "Hey there! I am very happy that you are happy ^_^  \n  \n  -Rapiro"


    header = "To: " + toAdd + "\n" + "From: " + fromAdd + "\n" + "Subject: " + subject
 

    #print header + "\n" + body

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()

    s.login(smtpUser, smtpPass)
    s.sendmail(fromAdd, toAdd, header + "\n" + body)

    s.quit()



# Cannot use due to memory error
def pca_calc(main):
    n_components = 90000
    print '----------------------'
    print main.shape
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(main)
    main = pca.transform(main)
    print main.shape
    return main

def motor_emotion_response(pred):
    if pred == 'Smiling':
        print 'Activating command. . .'
        os.system("./rapirocommands 6")
        sleep(5)
        os.system("./rapirocommands 0")
        print 'Command finished'
    elif pred == 'Neutral':
        print 'Activating neutral command. . .'
        os.system("./hellobash")
        sleep(5)
        os.system("./rapirocommands 5")
        sleep(5)
        os.system("./rapirocommands 0")
        print 'End command'
    elif pred == 'Angry':
        print 'Activating angry command. . .'
        os.system("./rapirocommands 4")
        sleep(2)
        os.system("./rapirocommands 0")
        print 'Command ended'
    elif pred == 'Shocked':
        print 'Activating shocked command'
        os.system("./rapiro-commands 2")
        sleep(2)
        os.system("./rapiro-commands 0")
        print 'Command ended'


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
    prev_gray = cv2.resize(prev_gray, (0,0), fx=0.27, fy=0.27)
    face = face_classifier.detectMultiScale(prev_gray, 1.2, 4)
    
    if len(face) == 0:
        print 'No face was detected'
        print prev_gray.shape
        exit()
    else:
        print 'Face detected'
        for (x,y,w,h) in face:
            prev_gray = prev_gray[y: y+h, x: x+w]

    capf.release()
    return (x,y,w,h, prev_gray)

def sensitive_override_check(prob_s, pred):
    if pred == 'Neutral':
        override_arr = [prob_s[0,3], prob_s[0,2], prob_s[0,0]]
        max_comp = max(override_arr)

        max_ind = [i for i, j in enumerate(override_arr) if j == max_comp][0]

        qualified_override = False
        if max_comp > 30:
            qualified_override = True

        if qualified_override:
            if max_ind == 0:
                pred = 'Smiling'
            elif max_ind == 1:
                pred = 'Shocked'
            else:
                pred = 'Angry'

        #print 'Sensitive Override triggered. . .'
    return pred

def emotion_to_speech(pred):
    engine = pyttsx.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)
    if pred == 'Neutral':
        speech = 'Hello, you seem fine today'
    elif pred == 'Smiling':
        speech = 'You seem happy. I am very happy that you are happy!'
    elif pred == 'Shocked':
        speech = 'What is wrong? You look like you seen a ghost.'
    elif pred == 'Angry':
        speech = 'Why are you angry? Did something annoy or frustrate you?'
    print speech 
    engine.say(speech)
    engine.runAndWait()



motor_emotion_response("Smiling")

    
slash = '/'

folder_trans = np.array([])
target = np.array([])
label_trans = np.array([])
folder = ''
choice = ''

face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

#print 'Load datasets [l] from file or create a new one [n]'
loading = 'l'

if loading == 'l':
    #print 'Press [p] to predict test dataset, or else press any key'
    predict_start = 'n'
else:
    predict_start = 'p'

if loading=='l':
    # load dataset matrix from npy file
    
    t0 = time()
    t1 = time()
    if predict_start == 'p':
        print 'Loading the main matrix. . .'
        
        main = np.load('optical-main-mini.npy')
        diff = diff = time() - t1
        print 'Loaded main matrix in ', diff, 's of size ', main.shape
        
        t2 = time()

        print 'Loading the target vector. . .'
        target = np.load('optical-target-mini.npy')
        diff = time() - t2
        print 'Loaded target in ', diff, 's of size ', target.shape
    
 
    

print 'Finished'
total_time = time() - t0
print total_time, 's'


t0 = time()
if loading == 'l':
    print 'Now loading trained model. . .'
    model = joblib.load('Optical-Model-Mini/optical-model-mini.pkl')
    t1 = time()
    print 'Loading time: ', round(time()-t0, 3), 's'

else:
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(main, target, test_size = 0.2)
    print 'Now training. . .'
    model = SVC(probability=True)

    '''
    #model = SVC(kernel='poly')
    #model = GaussianNB()
    '''
    model.fit(features_train, labels_train)
    print 'training time: ', round(time()-t0, 3), 's'
    print 'Saving model. . .'
    t1 = time()
    joblib.dump(model, 'Optical-Model-Mini/optical-model-mini.pkl')

    t3 = time()
    print 'model saving time: ', round(time()-t0, 3), 's'
print 'Now predicting. . .'




if predict_start == 'p':
    if loading == 'l':
        features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(main, target, test_size = 0.2)
    # Determine amount of time to train
    t1 = time()
    pred = model.predict(features_test)

    print 'predicting time: ', round(time()-t1, 3), 's'

    accuracy = accuracy_score(labels_test, pred)

    print 'Confusion Matrix: '
    print confusion_matrix(labels_test, pred)

    # Accuracy in the 0.9333, 9.6667, 1.0 range
    print accuracy




# ---------------------------------
while True:
    # Test with another video

    while True:
        print 'Press [n] to go into normal mode or [s] to go into sensitive mode'
        sensitive_out = raw_input()

        if sensitive_out == 'n' or sensitive_out == 's':
            break
    

    # Manually setting x, y, w, h values in order make more consistant test
    # and training videos
    
    
    x = 63
    y = 35
    w = 64
    h = 64

    #prev_gray = frame_f.copy()
    #prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)


    # Start video to record the user
    #cap to record user for 15 frames
    cap = cv2.VideoCapture(0)

    # Name of the video file
    path = 'test.h264'

    # Starting video
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path,fourcc, 20.0, (640,480))

    print 'Press any key to start recording'
    go = raw_input()


    # New recording feature for the Raspberry Pi
    
    with picamera.PiCamera() as camera:
        print 'Starting recording. . .'
        camera.vflip = True
        camera.start_recording(path)
        print 'Before sleep'
        sleep(5)
        print 'After sleep'
        print 'Stopping the camera from recording. . .'
        camera.stop_recording()
        print 'Finished recording'    
    
    # To get a
    # Cap3
    cap3 = cv2.VideoCapture(path)
    ret, prev_gray = cap3.read()
    prev_gray = cv2.cvtColor(prev_gray,cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (0,0), fx=0.27, fy=0.27)
    prev_gray = prev_gray[y: y+h, x: x+w]
    cap3.release()
 
    #face = face_classifier.detectMultiScale(prev_gray, 1.2, 4)
    
    j = 0
    # To analyze the recording and make an emotion prediction
    
    cap4 = cv2.VideoCapture(path)
    max_frame = 36
    while True:
        print 'j: ', j
        ret, frame = cap4.read()
        if frame == None:
            print 'Frame failure, trying again. . .'
            cap4.release()
            cap4 = cv2.VideoCapture(path)
            continue
        if j > max_frame + 1:
            cap4.release()
            break
        frame = cv2.resize(frame, (0,0), fx=0.35, fy=0.35)
        frame = frame[y: y+h, x: x+w]
        #cv2.imshow('To test with', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)

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

    cap4.release()
    #cv2.destroyAllWindows()

    print 'Now predicting. . .'
    
    ### Sliding window ###
    k_start = 0
    k_end = 15 * flow_mat.shape[0]

    max_frame = 36 * flow_mat.shape[0]
    while k_end < max_frame:
        count = float(k_end/max_frame)
        count = np.around(count, decimals=2)
        print count, '%'

        
        model.predict(sub_main[k_start:k_end])
        
        prob = model.predict_proba(sub_main[k_start:k_end])
        prob_s = np.around(prob, decimals=5)
        prob_s = prob_s* 100
        # Determine amount of time to predict
        t1 = time()
        pred = model.predict(sub_main[k_start:k_end])


        if sensitive_out == 's':
            pred = sensitive_override_check(prob_s, pred)

        if pred != 'Neutral':
            break

        
        k_start = k_start + (7 * flow_mat.shape[0])
        k_end = k_end + (7 * flow_mat.shape[0])

    ######################


    print 'predicting time: ', round(time()-t1, 3), 's'

    print ''
    print 'Prediction: '
    print pred

    print 'Probability: '
    print 'Neutral: ', prob_s[0,1]
    print 'Smiling: ', prob_s[0,3]
    print 'Shocked: ', prob_s[0,2]
    print 'Angry: ', prob_s[0,0]
    
  
    print 'Start hello 2'
    os.system("./hellobash")
    print 'End hello 2'



    emotion_to_text(pred)
    
    print 'Starting robot motion response'
    motor_emotion_response(pred)
    print 'Motion ended'
