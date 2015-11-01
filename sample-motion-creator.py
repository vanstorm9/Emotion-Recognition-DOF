from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
from time import sleep

def video_playback(path, prev_gray,x,y,w,h):
    cap4 = cv2.VideoCapture(path)
    j = 0
    while(cap4.isOpened()):
        ret, frame = cap4.read()

        if frame == None:
            print 'Frame failure, trying again. . .'
            cap4.release()
            cap4 = cv2.VideoCapture(path)
            continue
        
        '''
        if frame == None:
            cap4.release()
            break
        '''
        if j > 17:
            cap4.release()
            break
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        frame = frame[y: y+h, x: x+w]
        #cv2.imshow('To test with', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
        flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)

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

    cap4.release()
    
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

def sample_video_replay(frame, path):
    while True:
        print "Press [v] to view video, otherwise, press [n] to skip"
        view_video = raw_input()
        cv2.destroyAllWindows()
        while True:
            if view_video == 'v':
                while True:
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    prev_gray = cv2.resize(prev_gray, (0,0), fx=0.5, fy=0.5)
                    video_playback(path, prev_gray,x,y,w,h)
                    sleep(1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print 'You have quit'
                        break
                break
            else:
                break

        if view_video == 'n':
            break

face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

slash = '/'
underscore = '_'
dot = '.'
# Where to position the camera
print 'Automatic positioning [a] or manual position (default) [m]'
positioning = raw_input()

print 'Number of samples:'
iterations = raw_input()

print 'Number of frames per video:'
max_frames = raw_input()

print 'Name of folder to store:'
folder = raw_input()

print 'Name your file (without extension):'
file_name = raw_input()

print 'Name of extension:'
extension = raw_input()
print iterations

i = 0
j = 0

# Take inital frame
cap2 = cv2.VideoCapture(0)
if positioning == 'a':
    while True:
        ret_f, frame_f = cap2.read()
        face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        face = face_classifier.detectMultiScale(frame_f, 1.2, 4)

        if len(face) == 0:
            print "No face detected"
            print 'Press any key to try again'
            wait = raw_input()
            continue
        else:
            for (x,y,w,h) in face:
                frame_f = frame_f[y: y+h, x: x+w]
                break
            break
    
    cv2.imshow('First Frame', frame_f)
    cv2.waitKey(0)
else:
    # Manually setting x, y, w, h values in order make more consistant test
    # and training videos
    x = 118
    y = 66
    w = 116
    h = 116    

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




# Get all samples
while True:
    # Get user prepared
    print 'Press any key to record the next video'
    wait = raw_input()

    
    
    
    # Start video
    cap = cv2.VideoCapture(0)
    print 'i: ',i
    i_char = str(i)

    # Name of the video file
    path = folder + slash + file_name + underscore + i_char + dot + extension

    # Starting video
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path,fourcc, 20.0, (640,480))

    
    
    # get each frame per video
    while True:
        
        ret, frame = cap.read()

        if frame == None:
            print 're. . .'
            continue
        
        out.write(frame)
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        frame = frame[y: y+h, x: x+w]
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print 'You have quit'
            break
        
        # End of single sample video, save the video and move to next
        if j > int(max_frames):
            prev_gray = frame.copy()
            break
        print 'j: ', j
        j = j + 1


    cap.release()
    out.release()
    

    # View replay of video
    sample_video_replay(frame, path)
    
    print "Press 'd' to delete previous video, otherwise press another key to keep:"
    delete = raw_input()

    if delete != 'd':
        i = i + 1
    else:
        print 'Deleted!'
    j = 0
    # To escape main loop after all samples collected
    
    cv2.destroyAllWindows()
    
    if i >= int(iterations):
        break
    
    

    
    '''
    cv2.imshow('Video', old_frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(old_frame, 1.2, 4)

    if len(face) == 0:
        print "This is empty"
        continue
    else: 
        print 'Detected'
        for (x,y,w,h) in face:
            #focused_face = old_frame[y: y+h, x: x+w]
            focused_face = old_gray[y: y+h, x: x+w]
            #cv2.rectangle(old_frame, (x,y), (x+w, y+h), (0,255,0),2)

        #cv2.imshow('Detected_Frame', focused_face)
        #cv2.waitKey(0)
        i_char = str(i)
        path = folder + slash + file_name + underscore + i_char + dot + extension
        cv2.imwrite(path,focused_face)
        i = i + 1
    '''
print 'Finished!'
