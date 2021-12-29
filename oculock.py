# imports
import cv2
import os

from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from glob import glob

cascadePath = "./haarcascade_righteye_2splits.xml"
eyeCascade = cv2.CascadeClassifier(cascadePath)

def getEye(cam):
    # gets dimensions of webcam, used for calculating minimum size later
    cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # loops and checks frames for eye until eye is found
    while True:
        ret, frame = cam.read() # reads webcam
        
        # breaks loop if frame not found
        if not ret:
            print("failed grabbing frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscales image for easier eye-dentification
        
        eyes = eyeCascade.detectMultiScale (
            gray,
            scaleFactor = 1.1,  
            minNeighbors = 60,  
            minSize = (cam_width//7, cam_height//7),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # returns cropped frame of eye for use
        if (len(eyes) > 0):
            for (x, y, w, h) in eyes:
                cropped_eye = frame[y:y+h, x:x+w]
            return cropped_eye
    return None # returns none if camera dont work

def makeDataset(dataset_size = 175):
    cam = cv2.VideoCapture(0)
    
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
    
    while getEye(cam) is not None:
        count+=1
        frame = getEye(cam) # gets current frame containing an eye
        # grayscales eye and resizes for consistency 
        dataset_eye = cv2.resize(frame, (224, 224))
        dataset_eye = cv2.cvtColor(dataset_eye, cv2.COLOR_BGR2GRAY)
        # writes image of eye into specified directory
        if count < dataset_size//3:
            c_dir = test_dir
        else:
            c_dir = training_dir
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

    # constant containing our image size: used to resize images later
    IMAGE_SIZE = [224, 224]
  
    # adds preprocessing layer to front of VGG
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    # tells program to not train existing weights
    for layer in vgg.layers:
        layer.trainable = False
    # gets classes in both train and test folders
    train_folders = glob('./train/images/*')
    print(len(train_folders))
    test_folders = glob('./test/images/*') 
    # our layers
    x = Flatten()(vgg.output)
    x = Dense(1000, activation='relu')(x)
    x = Dropout (0.5)(x)
    x = Dense(1000, activation='relu')(x)
    
    prediction = Dense(len(train_folders), activation='softmax') (x)
    
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
    
    nb_train_samples = len(train_folders)
    nb_validation_samples = len(test_folders)
    batch_size = 16
    # fit the model
    r = model.fit(    
        training_set,
        validation_data=test_set,
        epochs=5,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_steps = nb_validation_samples // batch_size)
    
    model.save("./models/" + name + "_eye_new_model.h5")