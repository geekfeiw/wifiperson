#!/usr/bin/python

import cv2
import datetime
import time
#import sys


if __name__ == "__main__":

    try:
    
        fps = 20
        frameWidth  = 1280
        frameHeight = 720
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frameWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
#	time.sleep()
        cap.set(cv2.CAP_PROP_FPS, fps)

	cameraFPS = cap.get(cv2.CAP_PROP_FPS)

	print("FPS:", cameraFPS)
	print("Frame size:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG') # + .avi works, .mp4 not works
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')MP4V
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        videofile = cv2.VideoWriter('video.avi',
                                    fourcc,
                                    int(cameraFPS),
                                    (frameWidth, frameHeight))
        
        
        
        #file = open('/media/csipose1/XPG SD700X/time', 'w+')
        
        
        with open('VideoTimestamp.txt', 'w+') as file:
            while(cap.isOpened()):
                ret, frame = cap.read()
                #time.sleep(delay)
                t = datetime.datetime.now()
                #t = time.clock()
                #print(ret)
                if ret:
                   file.write(str(t)+'\n')
                   print(str(t))
                   videofile.write(frame)
                   # cv2.imshow('Camera', frame)
                    
                   #if cv2.waitKey(1) & 0xFF == ord('q'):
                   #    break
                else:
                    break
    
        
    except KeyboardInterrupt:
        print("Quit")
        cap.release()
        videofile.release()
        #cv2.destroyAllWindows()
        file.close()
