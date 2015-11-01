import cv2
import os
#path = "datasets/anthony-6-6-15_9.avi"

underscore = '_'
extension = '.jpg'
location = 'test/'
num_of_frames = 17
while True:
    print 'What video do you want to dissect?'
    path = raw_input()

    print 'What will be the name of the set of frames?'
    name_frame = raw_input()

    '''
    path = 'datasets/Subjects/Chuah/Angry/test_0_14.avi'
    name_frame = 'testing'
    '''

    cap = cv2.VideoCapture(path)
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

        h = frame.shape[0]
        w = frame.shape[1]
        '''
        tra_frame = frame[0:h , 0:w]
        frame[0:h - 10 , 0:w] = tra_frame
        '''
        '''
        tra_frame = frame[280:340, 330:390]
        frame[273:333, 100:160] = tra_frame
        '''
        # y_trans = -147
        y_trans = 0
        y_1 = y_trans*-1
        x_trans = 0
        x_1 = x_trans*-1
        
        # [y1:y2 , x1:x2]
        tra_frame = frame[y_1:h, x_trans:w]
        frame[0:h + y_trans , 0:w + x_1] = tra_frame

        cv2.imshow('frame', frame)

        pic_name = location + name_frame + underscore + str(i) + extension
        
        cv2.imwrite(pic_name, frame)
        i = i + 1
    cap.release()
    print 'i: ' + str(i)

    while True:
        print 'Sew all frames into a video? Press [y] or [n]'
        sew_command = raw_input()
        if sew_command == 'y' or sew_command == 'n':
            break


    while True:
        print 'Replace original video? Press [r] or [n]'
        replace_vid = raw_input()
        if replace_vid == 'r' or replace_vid == 'n':
            if replace_vid == 'n':
                path = 'test/test_sample.avi'
            break


    # Sew the videos together
    if sew_command == 'y':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path,fourcc, 20.0, (640,480))

        i = 0
        while True:

            if i > num_of_frames:
                break
            
            pic_name = location + name_frame + underscore + str(i) + extension
            frame = cv2.imread(pic_name)
            # Convert to grayscale
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            if frame == None:
                print 'Frame failure, trying again. . .'
                continue
            
            print i

            out.write(frame)
            i = i + 1
            os.remove(pic_name)
        out.release()
    print 'Done!'
    print ''
    print ''
