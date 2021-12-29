# imports
import cv2
import numpy
import os

cascadePath = "./haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cascadePath)

def getEye(cam):
    
    # gets dimensions of webcam, used for calculating minimum size later
    cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # loops and checks frames for eye until eye is found
    while True:
        ret, frame = cam.read() # reads webcam
        
        # breaks loop if frame not found
        if ret is None:
            print("failed grabbing frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
        
        eyes = eyeCascade.detectMultiScale (
            gray,
            scaleFactor = 1.1,  
            minNeighbors = 80,  
            minSize = (cam_width//6, cam_height//6),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # returns cropped frame of eye for use
        if (len(eyes) > 0):
            for (x, y, w, h) in eyes:
                cropped_eye = frame[y:y+h, x:x+w]
            return cropped_eye
    return None # returns none if camera dont work

def makeDataset(dataset_size):
    cam = cv2.VideoCapture(0) # initializes webcam
    training_dir = "./train/" # directory containing images to be trained
    
    # creates a new directory if training_dir not found
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
    
    count = 0
    # collects images of eyes to train later
    while ocutil.getEye(cam) is not None:
        count+=1
        frame = ocutil.getEye(cam) # gets current frame containing an eye
        # grayscales eye and resizes for consistency 
        dataset_eye = cv2.resize(frame, (224, 224))
        dataset_eye = cv2.cvtColor(dataset_eye, cv2.COLOR_BGR2GRAY)
        cv2.imwrite((training_dir +  str(count) + ".jpg"), dataset_eye) # writes image of eye into directory
        print("saved image", count, "into", training_dir)
        if count >= dataset_size:
            print("made dataset")
            break