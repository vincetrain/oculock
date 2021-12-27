# imports
import cv2
import numpy as np


def scanEyes():
    cam = cv2.VideoCapture(0) # setup camera
    w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # creates cascade from provided cascade file
    cascPath = "haarcascade_eye.xml"
    eyeCascade = cv2.CascadeClassifier(cascPath)

    # reads camera for available faces
    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame, (1280, 720))   # resizes frame to 
        # frame = cv2.flip(frame, 1)
        if not ret:
            print("failed grabbing frame")
            break
        
        key = cv2.waitKey(1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        eyes = eyeCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 40,
            minSize = (int(w/7), int(h/7)),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("test", frame)
        
        if (len(eyes) > 0):
            cv2.waitKey(0)
            if (key == 27):
                break

    cam.release()
    cv2.destroyAllWindows()
    
    
def calculateWinDimension(x):
    while x > 512:
        x-=5
    return int(x),

scanEyes()