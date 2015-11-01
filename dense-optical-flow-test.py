from matplotlib import pyplot as plt
import numpy as np
import math
import cv2

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


face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture(0)

x = 0
y = 0
w = 0
h = 0

print 'Test'
# To keep taking pictures until a face is found
while True:
    ret, old_frame = cap.read()

    prev_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('test', prev_gray)
    cv2.waitKey(0)

    face = face_classifier.detectMultiScale(old_frame, 1.2, 4)

    print face

    if len(face) == 0:
        print "No face detected"
        continue

    for (x,y,w,h) in face:
        prev_gray = prev_gray[y: y+h, x: x+w]
        #prev_gray = cv2.resize(prev_gray, (0,0), fx=0.5, fy=0.5)
        #cv2.rectangle(old_frame, (x,y), (x+w, y+h), (0,255,0),2)
    break
i = 0

# Open VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('datasets/test.avi',fourcc, 20.0, (640,480))

if (not out.isOpened()):
    print "Video writer error"

cv2.imshow('test', prev_gray)
cv2.waitKey(0)


# To open the video with dense optical flow
while True:

    #print i
    ret, old_frame = cap.read()
    out.write(old_frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        print 'You are quitting now'
        break
    gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    gray = gray[y: y+h, x: x+w]
    #gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
    flow= cv2.calcOpticalFlowFarneback(prev_gray,gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = gray
    '''
    print status
    print ''
    print ''
    print ''
    '''
    cv2.imshow('Optical flow',draw_flow(gray,flow))
    '''
    if i > 40:
        break
    '''
    i = i + 1

out.release()
cv2.destroyAllWindows()
print 'Finished!'


