from flask import Flask, render_template, Response
import cv2
# import oculock as ocu

import os
import cv2
import numpy as np

import jyserver.Flask as jsf

# from PIL import Image
# from glob import glob
# from keras.models import load_model

app = Flask(__name__)
TEMPLATES_AUTO_RELOAD = True
camera= cv2.VideoCapture(1)
detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("./haarcascade_eye.xml")

def generate_frames():
    while True:
        ## read the camera frame
        success,frame=camera.read()
        frame = cv2.flip(frame, 1)  
        if not success:
            break
        else:
            gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY) 
            faces=detector.detectMultiScale(frame, 1.1,20)
        #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eyeCascade.detectMultiScale(roi_gray, 1.3, 5)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
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

# @app.route('/compare_eye')
# def compare_eye():
#     '''
#     TODO:
#     - Call eye_extract
#     - Compare returned cropped eye with pretrained model(s)
#     - If eye is found within pretrained model/passes check, return HTML element containing "passed " and a green box.
#         (the css elements dont have to be set using style here, just have it return a seperate id, aka passed or unpassed or something)
#     - Else, return html element containing "not passed" or something and a red box.
#     '''
#     status = "UNLOCKED"
#     return render_template('index.html', status=status)

@app.route('/')
def rec_status():
    status = "LOCKED"

    return render_template('index.html', status=status)

@app.route('/video')
def video():    
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')




# def trainModel():
#     name = str(input("Enter name for model: "))
#     cv2.namedWindow("Cropped Eye")
#     ocu.makeDataset()
#     ocu.makeModel(name)
#     pass

# @app.route('/eye_recog')
# def eye_recog():
#     if not os.path.exists("./models/"):
#         print("No models available. Try training a model first.")
#         return
    
#     model_list = glob("./models/*")
    
#     print("Model List")
#     i = 0
    
#     for n in model_list:
#         print(str(i+1) + ")" , n)
#         i += 1
        
#     userInput = 0
#     while userInput not in range(1, len(model_list)+1):
#         userInput = int(input("Select desired model: "))
        
#     desired_model = load_model(model_list[userInput - 1])
    
#     cam = cv2.VideoCapture(1)
#     while True:
#         _, frame = cam.read()
        
#         eye = eye_extract(cam, frame)
        
#         if type(eye) is np.ndarray:
#             eye = cv2.resize(eye, (224, 224))
#             img = Image.fromarray(eye, "RGB")
            
            
#             img_array = np.array(img)
            
#             img_array = np.expand_dims(img_array, axis=0)
#             pred = desired_model.predict(img_array)
            
#             print(pred)
            
#             text = "locked"
#             if (pred[0][0] > 0.9):
#                 text = "unlocked"
#             cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
#         else:
#             cv2.putText(frame, "no eye found", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
#         cv2.imshow("Eye Recognition Demonstration", frame)
#         key = cv2.waitKey(1)
#         if key == 27:
#             break
#     cam.release()
#     cv2.destroyAllWindows()

if __name__=="__main__":
    app.run(debug=True)

