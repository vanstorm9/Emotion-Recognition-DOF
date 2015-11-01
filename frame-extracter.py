import cv2

#path = "datasets/anthony-6-6-15_9.avi"

underscore = '_'
extension = '.jpg'
location = 'test/'
num_of_frames = 17

print 'What video do you want to dissect?'
path = raw_input()

print 'What will be the name of the set of frames?'
name_frame = raw_input()

cap = cv2.VideoCapture(path)
'''
while not cap.isOpened():
    cap = cv2.VideoCapture(path)
    cv2.waitKey(1000)
    print "Wait for the header"
'''
#pos_frame = cap.get(cv2.CV_CAP_PROP_POS_FRAMES)
i = 0
while cap.isOpened():
    flag, frame = cap.read()
    # Convert to grayscale
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if frame == None:
        print 'Frame failure, trying again. . .'
        cap.release()
        cap = cv2.VideoCapture(path)
        continue
    if i > num_of_frames:
        break
    print i 
    cv2.imshow('frame', frame)

    pic_name = location + name_frame + underscore + str(i) + extension
    
    cv2.imwrite(pic_name, frame)
    i = i + 1
cap.release()
print 'i: ' + str(i)
