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
    while getEye(cam) is not None:
        count+=1
        frame = getEye(cam) # gets current frame containing an eye
        # grayscales eye and resizes for consistency 
        dataset_eye = cv2.resize(frame, (224, 224))
        dataset_eye = cv2.cvtColor(dataset_eye, cv2.COLOR_BGR2GRAY)
        cv2.imwrite((training_dir +  str(count) + ".jpg"), dataset_eye) # writes image of eye into directory
        print("saved image", count, "into", training_dir)
        if count >= dataset_size:
            print("made dataset")
            break

def makeModel ():
    from PIL import Image
    from keras.layers import Input, Lambda, Dense, Flatten, Dropout
    from keras.models import Model
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import RMSprop
    from keras.models import Sequential
    import numpy as np
    from glob import glob

    # re-size alLl the images to this
    IMAGE_SIZE = [224, 224]
    train_path = 'src/images/'
    valid_path = 'src/sure/'
    # add preprocessing layer to the front of VGG
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False
    # useful for getting number of classes
    folders = glob('src/images/*') 
    # our layers
    x = Flatten()(vgg.output)
    x = Dense(1000, activation='relu')(x)
    x = Dropout (0.5)(x)
    x = Dense(1000, activation='relu')(x)
    #x = Dropout (0.5)(x)
    prediction = Dense(len(folders), activation='softmax') (x)
    
    # create a model object
    model = Model(inputs=vgg.input, outputs=prediction)
    # view the structure of the model
    print(model.summary())
    # tell the model what cost and optimization method to use
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate = 0.001),
        metrics=['accuracy']
    )
    
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator (rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True) 
    test_datagen = ImageDataGenerator (rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('src/images/',
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('src/sure/',
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical') 
    # Enter the number of training and validation samples here
    
    nb_train_samples = 100
    nb_validation_samples = 50
    batch_size = 16
    # fit the model
    r = model.fit(    
        training_set,
        validation_data=test_set,
        epochs=5,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_steps= nb_validation_samples // batch_size)
    
    from keras.models import load_model
    model.save('facefeatures_new_model.h5')
                                                         
def compareEyes(img):
    ## TODO: COMPARE GIVEN EYE FRAME WITH TRANSFER-LEARNED MODEL
    from PIL import Image as Img
    from PIL import ImageTk
    from keras.applications.vgg16 import preprocess_input
    import base64
    from io import BytesIO
    import json
    import random
    from keras.models import load_model
    from keras.preprocessing import image
    #Loading the cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #Function detects faces and returns the cropped face
    #If no face detected it returns the input image
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # if faces is ():
    #     return None
    #Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255,255),2)
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face

def pseudoFrontEnd (model_dir, cam):
    from PIL import Image as Img
    from PIL import ImageTk
    from keras.applications. vgg16 import preprocess_input
    import base64
    from io import BytesIO
    import json
    import random
    import cv2
    from keras.models import load_model
    import numpy as np
    from keras.preprocessing import image
        
    while True:
        _, frame = cam.read()
        
        model = load_model(model_dir)
            
        face = getEye(cam)
        if type(face) is np.ndarray:
            face=cv2.resize(face, (224,224))
            im= Img.fromarray(face , 'RGB')
            #Resizing because we trained the model with this size
            img_array = np.array(im)
            #Keras model used 4D tensor so we change the dimension from 128x128x3 to 1x128x128x3
            img_array = np.expand_dims (img_array, axis=0)
            pred = model.predict(img_array)
            print(pred)
            name= "It's not you"
            #name=""
            if(pred[0][0]>0.9):
                name = "annie"
            #cv2. put Text(frame, name, (50,50), cv2.FONT HERSHEY COMPLEX, 1, (e,255,0), 2)
            cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
        else:
            cv2.putText(frame, "No Face Found", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            cv2.imshow('Video', frame)
        #if cv2.waitKey(1) & @XFF == ord('q'):
        key = cv2.waitKey(1)
        if key == 27:
                break  
    