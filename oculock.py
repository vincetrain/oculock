# imports
from __future__ import print_function
from keras import backend as K
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.models import Model
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from glob import glob

import keras
import cv2
import os



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
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
    
    eyes = eyeCascade.detectMultiScale(gray, 1.3, 5)
    
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
            cropped_eye = gray[y:y+h, x:x+w]
        return cropped_eye

def makeDataset():
    cam = cv2.VideoCapture(1)
    dataset_size = 150
    
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
            dataset_eye = cv2.cvtColor(dataset_eye, cv2.COLOR_BGR2GRAY)
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

    # creates a models path if not already exists
    if not os.path.exists("./models"):
        os.mkdir("./models")

    #mini batch gradient descent ftw
    batch_size = 20
    #10 difference characters
    num_classes = 10
    #very short training time
    epochs = 5

    # input image dimensions
    #28x28 pixel images. 
    img_rows = 224
    img_cols = 224

    # the data downloaded, shuffled and split between train and test sets
    #if only all datasets were this easy to import and format
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)  
    print(x_train.size)     
    print(len(x_train))   

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    #more reshaping
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    from keras.utils.np_utils import to_categorical
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    #build our model
    model = Sequential()
    #convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    #again
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    #flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    #fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    #one more dropout for convergence' sake :) 
    model.add(Dropout(0.5))
    #output a softmax to squash the matrix into output probabilities
    model.add(Dense(num_classes, activation='softmax'))
    #Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
    #categorical ce since we have multiple classes (10) 
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=RMSprop(learning_rate = 0.001),
                metrics=['accuracy'])

    #train that ish!
    r = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
        
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.plot(r.history['loss'], label = 'loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.title("Loss vs Val_Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    model.save("./models/" + name + "_eye_new_model.h5")