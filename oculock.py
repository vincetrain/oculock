# imports
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten, Dropout
from keras. models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from glob import glob

cascadePath = "./haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cascadePath)

# creates directories for storing the train and test images
if not os.path.exists("./train"):
    os.mkdir("./train")
if not os.path.exists("./test"):
    os.mkdir("./test")

def getEye(cam, frame):
    # gets dimensions of webcam, used for calculating minimum size later
    cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
    
    eyes = eyeCascade.detectMultiScale(frame, 1.3, 5)
    
    # eyes = eyeCascade.detectMultiScale (
    #     gray,
    #     scaleFactor = 1.1,  
    #     minNeighbors = 30,  
    #     minSize = (cam_width//7, cam_height//7),
    #     flags = cv2.CASCADE_SCALE_IMAGE
    # )
    
    # returns cropped frame of eye for use
    if (len(eyes) > 0):
        for (x, y, w, h) in eyes:
            cropped_eye = frame[y:y+h, x:x+w]
        return cropped_eye

def makeDataset():
    cam = cv2.VideoCapture(1)
    dataset_size = 1050
    
    training_dir = "./train/images/" # directory containing images to be trained
    test_dir = "./test/images/" # directory containing images to be trained
    
    c_dir = ""
    
    # creates a new directory if training_dir not found
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
        
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    count = 0
    # collects images of eyes to train later
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("unable to get frame")
            break
        if getEye(cam, frame) is not None:
            count+=1
            eye = getEye(cam, frame) # gets current frame containing an eye
            # grayscales eye and resizes for consistency 
            dataset_eye = cv2.resize(eye, (224, 224))
            # dataset_eye = cv2.cvtColor(dataset_eye, cv2.COLOR_BGR2GRAY)
            # writes image of eye into specified directory
            c_dir = training_dir
            if count <= 50:
                c_dir = test_dir       
            cv2.imwrite((c_dir +  str(count) + ".jpg"), dataset_eye)
            
            print("saved image", count, "into", training_dir)
            
            # displays current cropped eye in window
            cv2.putText(dataset_eye, str(count), (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
            cv2.imshow("Cropped Eye", dataset_eye)
        key = cv2.waitKey(1)
        if key == 27 or count >= dataset_size:
            break
        
    cam.release()
    cv2.destroyAllWindows()
        
def makeModel(name):
    # re-size all the images to this
    IMAGE_SIZE = [224, 224] 
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False 
    
    folders = glob('./train/*') 

    # our layers
    x = Flatten()(vgg.output)
    x = Dense (1000, activation='relu')(x)
    x = Dropout (0.5)(x)
    x = Dense(1000, activation='relu')(x)
    prediction = Dense (len(folders), activation='softmax') (x)

    model = Model(inputs=vgg.input, outputs=prediction)

    # view the structure of the model
    print(model.summary())
    # tell the model what cost and optimization method to use
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate = 0.0001),
        metrics=['accuracy']
    ) 
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True) 
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('./train/',
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('./test/',
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical')
    
    print(training_set.class_indices)

    # fit the model
    r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch= 32,
    validation_steps= 6)
    
    plt.style.use('ggplot')
    plt.plot(r.history['loss'], label = 'loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.title("Loss vs Val_Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    model.save("./models/" + name + "_eye_new_model.h5")