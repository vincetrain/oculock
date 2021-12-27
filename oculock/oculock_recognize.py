# imports
import cv2
import os

def readCam(cam):
    ret, frame = cam.read() # reads webcam
            
    # returns frame if frame is found
    if ret:
        return frame
    
    # returns None and tells user if frame not found
    print("failed grabbing frame")  
    return None
        

def getEye(cam, eyeCascade):
    # gets dimensions of webcam, used for calculating minimum size later
    cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # loops infinitely while displaying frames of current webcam
    while readCam(cam) is not None:
        frame = readCam(cam)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
        
        eyes = eyeCascade.detectMultiScale (
            gray,
            scaleFactor = 1.1,  
            minNeighbors = 80,  
            minSize = (cam_width//6, cam_height//6),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # freezes and returns frame for use if an eye is found
        if (len(eyes) > 0):
            for (x, y, w, h) in eyes:
                cropped_eye = frame[y:y+h, x:x+w]
            return cropped_eye
        return None

def captureSamples(max_samples):
    pass

if __name__ == "__main__":
    cam = cv2.VideoCapture(0) # sets up camera

    cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cascPath = "haarcascade_eye.xml"    # dir to cascade file, DO NOT CHANGE UNLESS NEEDED
    eyeCascade = cv2.CascadeClassifier(cascPath)    # link to cascade file
    
    max_samples = 75 # max sample size
    count = 0 # counter variable for samples
    training_dir = "train/" # directory of where training images should go
    
    # creates a new directory 
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
    
    # detects and highlights eyes
    cv2.namedWindow("Oculock | Cropped Eye")
    while readCam(cam) is not None:
        raw = readCam(cam)
        frame = cv2.resize(raw, (int(cam_width*.75), int(cam_height*.75)))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
        
        cropped_eye = getEye(cam, eyeCascade)
        
        eyes = eyeCascade.detectMultiScale (
            gray,
            scaleFactor = 1.1,  
            minNeighbors = 80,  
            minSize = (cam_width//8, cam_height//8),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in eyes:
             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # creates a new window containing frame
        cv2.imshow("Oculock | Raw Camera", frame)
        
        if (cropped_eye is not None):
            if (count < max_samples):
                count += 1
                # grayscales eye and resizes for consistency 
                dataSetEye = cv2.resize(cropped_eye, (224, 224))
                dataSetEye = cv2.cvtColor(dataSetEye, cv2.COLOR_BGR2GRAY)
                
                # saves training images into a specified directory with unique names
                cv2.imwrite((training_dir +  str(count) + ".jpg"), dataSetEye)
                cv2.putText(cropped_eye, str(count), (50,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0),2)
            else:
                cv2.putText(cropped_eye, "capture complete", (50,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0),2)
                
            cv2.imshow("Oculock | Cropped Eye", cropped_eye)
            
        key = cv2.waitKey(1)
        
        # closes window if ESCAPE key is pressed
        if key == 27:
            break
    
    cam.release()
    cv2.destroyAllWindows()