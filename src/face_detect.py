# imports
import cv2
import numpy as np

cam = cv2.VideoCapture(0) # setup camera

# creates cascade from provided cascade file
cascPath = "haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cascPath)


ret, frame = cam.read()
# reads camera for available faces
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
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
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("test", frame)
        
    if (key == ord("q")):
        break

cam.release()
cv2.destroyAllWindows()