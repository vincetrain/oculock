# imports
import cv2
import numpy as np

def scanEyes(cam, eyeCascade):
    
    # gets dimensions of webcam, used for calculating minimum size later
    w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # loops infinitely while displaying frames of current webcam
    while True:
        ret, frame = cam.read() # reads webcam
        frame = cv2.resize(frame, (1280, 720))   # resizes frame to 1280x720 (16:9 res)
        frame = cv2.flip(frame, 2) # mirrors webcam (1 mirrored, 2 un-mirrored)
        
        # breaks loop and closes windows if frame was unable to be retrieved
        if not ret:
            print("failed grabbing frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
        
        # scans frame for any eyes
        eyes = eyeCascade.detectMultiScale(
            gray,   # takes grayscaled image
            scaleFactor = 1.1,  # scales something, idk
            minNeighbors = 40,  # margin of error, probably
            minSize = (int(w/7), int(h/7)), # determines minimum size eye
            flags = cv2.CASCADE_SCALE_IMAGE # references our cascade provided
        )
        
        # draws a rectangle around any apparent eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.imshow("test", frame) # updates window with new frame
        
        key = cv2.waitKey(1)
        
        # freezes and returns frame for use if an eye is found
        if (len(eyes) > 0):
            return frame
        
        # closes window if escape key is pressed
        if (key == 27):
            break

if __name__ == "__name__":
    cam = cv2.VideoCapture(0) # sets up camera

    cascPath = "haarcascade_eye.xml"    # dir to cascade file, DO NOT CHANGE UNLESS NEEDED
    eyeCascade = cv2.CascadeClassifier(cascPath)    # link to cascade file

    eyeFrame = scanEyes(cam, eyeCascade)
    
    