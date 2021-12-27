# imports
import cv2

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
    cam_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # loops infinitely while displaying frames of current webcam
    while readCam(cam) is not None:
        frame = readCam(cam)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
        
        # detects any eyes within frame using provided cascade
        eyes = eyeCascade.detectMultiScale (
            gray,
            scaleFactor = 1.1,  
            minNeighbors = 40,  
            minSize = (int(cam_width/8), int(cam_height/8)),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # freezes and returns frame for use if an eye is found
        if (len(eyes) > 0):
            for (x, y, w, h) in eyes:
                cropped_eye = frame[y:y+h, x:x+w]
            return cropped_eye
        return None
  
def getSamples(cam, eyeCascade, sample_size=75):
    count = 0
    
    frame = getEye(cam, eyeCascade)
    
    # captures 75 sample images of eye
    while True:
        
        if frame is not None:
            count += 1

            # grayscales eye and resizes for consistency 
            dataSetEye = cv2.resize(frame, (224, 224))
            dataSetEye = cv2.cvtColor(dataSetEye, cv2.COLOR_BGR2GRAY)
            
            # saves training images into a specified directory with unique names
            training_dir = 'src/train/images/' + str(count) + '.jpg'
            cv2.imwrite(training_dir, dataSetEye)
            print("saved to ", training_dir)

        if cv2.waitKey(1) == 27 or count == sample_size:
            break

if __name__ == "__main__":
    cam = cv2.VideoCapture(0) # sets up camera

    cam_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cascPath = "haarcascade_eye.xml"    # dir to cascade file, DO NOT CHANGE UNLESS NEEDED
    eyeCascade = cv2.CascadeClassifier(cascPath)    # link to cascade file

    # detects and highlights eyes
    cv2.namedWindow("Oculock | Cropped Eye")
    while readCam(cam) is not None:
        cropped_eye = getEye(cam, eyeCascade)
        frame = readCam(cam)
        frame = cv2.resize(frame, (int(cam_width*.75), int(cam_height*.75)))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
        
        # detects any eyes within frame using provided cascade
        eyes = eyeCascade.detectMultiScale (
            gray,
            scaleFactor = 1.1,  
            minNeighbors = 80,  
            minSize = (30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # draws a rectangle around detected eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # creates a new window containing frame
        cv2.imshow("Oculock | Raw Camera", frame)
        
        if (cropped_eye is not None):
            cv2.imshow("Oculock | Cropped Eye", cropped_eye)
            
        
        key = cv2.waitKey(1)
        
        # closes window if ESCAPE key is pressed
        if key == 27:
            break
    
    cam.release()
    cv2.destroyAllWindows()