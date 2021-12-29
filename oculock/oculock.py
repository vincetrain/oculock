# imports
import cv2
import os

from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from glob import glob

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
    test_dir = "./test/" # directory containing images to be trained
    
    # creates a new directory if training_dir not found
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
        
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    count = 0
    # collects images of eyes to train later
    while getEye(cam) is not None:
        count+=1
        frame = getEye(cam) # gets current frame containing an eye
        # grayscales eye and resizes for consistency 
        dataset_eye = cv2.resize(frame, (224, 224))
        dataset_eye = cv2.cvtColor(dataset_eye, cv2.COLOR_BGR2GRAY)
        # writes image of eye into specified directory
        if count < dataset_size//3:
            cv2.imwrite((test_dir +  str(count) + ".jpg"), dataset_eye) 
        else:
            cv2.imwrite((training_dir +  str(count) + ".jpg"), dataset_eye)
        print("saved image", count, "into", training_dir)
        if count >= dataset_size:
            print("made dataset")
            break

def makeModel (dataset_size):

    # re-size alLl the images to this
    IMAGE_SIZE = [224, 224]
  
    # add preprocessing layer to the front of VGG
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False
    # useful for getting number of classes
    folders = glob('./train/*') 
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
    
    training_set = train_datagen.flow_from_directory('./train/',
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('./test/',
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical') 
    # Enter the number of training and validation samples here
    
    # TODO: un-fix sample size
    nb_train_samples = dataset_size//3
    nb_validation_samples = 50
    batch_size = 16
    # fit the model
    r = model.fit(    
        training_set,
        validation_data=test_set,
        epochs=5,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_steps= nb_validation_samples // batch_size)
    
    model.save('eyefeatures_new_model.h5')
                                                         
def compareEye(img):
    ## TODO: COMPARE GIVEN EYE FRAME WITH TRANSFER-LEARNED MODEL
    
    #Loading the cascades
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_default.xml')

    #Function detects eyes and returns the cropped eye
    #If no eye detected it returns the input image
    eyes = eye_cascade.detectMultiScale(img, 1.3, 5)

    # 
    for (x,y,w,h) in eyes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255,255),2)
        cropped_eye = img[y:y+h, x:x+w]
        return cropped_eye