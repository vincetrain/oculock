import oculock as ocu

import os
import cv2
import numpy as np

from PIL import Image
from glob import glob
from keras.models import load_model

eyeCascade = cv2.CascadeClassifier("./haarcascade_eye.xml")

def eye_extract(cam, frame):
    cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # detects all right eyes within frame using eyeCascade
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    eyes = eyeCascade.detectMultiScale (
        gray,
        scaleFactor = 1.1,  
        minNeighbors = 30,  
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    # returns none if eyes not found
    if eyes == ():
        return None
    
    # highlights and crops eye
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,255),2)
        cropped_eye = frame[y:y+h, x:x+w]
    
    return cropped_eye

def trainModel():
    name = str(input("Enter name for model: "))
    cv2.namedWindow("Cropped Eye")
    ocu.makeDataset()
    ocu.makeModel(name)
    pass

def eye_recog():
    if not os.path.exists("./models/"):
        print("No models available. Try training a model first.")
        return
    
    model_list = glob("./models/*")
    
    print("Model List")
    i = 0
    
    for n in model_list:
        print(str(i+1) + ")" , n)
        i += 1
        
    userInput = 0
    while userInput not in range(1, len(model_list)+1):
        userInput = int(input("Select desired model: "))
        
    desired_model = load_model(model_list[userInput - 1])
    
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        
        eye = eye_extract(cam, frame)
        
        if type(eye) is np.ndarray:
            eye = cv2.resize(eye, (224, 224))
            img = Image.fromarray(eye, "RGB")
            
            
            img_array = np.array(img)
            
            img_array = np.expand_dims(img_array, axis=0)
            pred = desired_model.predict(img_array)
            
            print(pred)
            
            text = "not passed"
            if (pred[0][0] > 0.9):
                text = "passed"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "no eye found", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
        cv2.imshow("Eye Recognition Demonstration", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
        
while True:
    print("Oculock Demonstration\n"+
        "1) Train a model\n" +
        "2) Demonstrate eye-recognition\n" +
        "9) Exit")
    userInput = str(input("Please select an option: "))
    
    if userInput == "1":
        trainModel()
    elif userInput == "2":
        eye_recog()
    elif userInput == "9":
        break

exit()
        



