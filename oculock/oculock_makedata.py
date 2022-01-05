# imports
import cv2
import oculock_util as ocutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from mtcnn import *

from keras.layers import Dropout, Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.models import Model
from keras.models import Sequential

from glob import glob

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract_face(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face

        
# def extract_face(frame):
#     '''
#     Extracts faces from given frame into a Numpy array
    
#     Converts image into a Numpy array, where MTCNN weights are used to detect a face
#     and upon detection, frame is cropped and resize to display only the face in a 224x224 image.
#     The cropped face is then converted to a Numpy array and returned.
    
#     Returns None if no results are found
#     '''

#     detector = MTCNN()
#     results = detector.detect_faces(frame)
    
#     if results is None:
#         return None
#     x1, y1, width, height = results[0]['box']
#     x2, y2 = x1 + width, y1 + height
    
#     face = frame[y1:y2, x1:x2]
    
#     image = Image.fromarray(face)
#     # image = image.resize((224, 224))
    
#     return face

def make_data(cam, name):
    '''
    Extracts 200 images, distrubuting between testing and training images
    following the 80:20 rule, where 40 images are stored as test images,
    and 160 are stored as training images.
    
    Extracts face from extract_face(frame) function, where it is then converted
    to a grayscale image using cv2.COLOR_BGR2GRAY and input into either the test_dir or
    training_dir, dependant on how many images have already been taken
    '''
    dataset_size = 200
    
    training_dir = f'train/face/{name}/' 
    test_dir = f'test/face/{name}/' 
    
    ocutil.mkdir(training_dir)
    ocutil.mkdir(test_dir)
    
    count = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error occured while grabbing frame.")
            break
        raw_face = extract_face(frame)
        if raw_face is not None:
            count+=1
            face = cv2.cvtColor(raw_face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(raw_face, (224, 224))
            c_dir = training_dir
            if count <= 40:
                c_dir = test_dir       
            cv2.imwrite((c_dir + str(count) + ".jpg"), face)
            print("saved image", count, "into", c_dir)
        if count >= dataset_size:
            break

def make_model2(name):
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
    #import matplotlib.pyplot as plt

    # re-size all the images to this
    IMAGE_SIZE = [224, 224]

    train_path = 'train/face/'
    valid_path = 'test/face/'

    # add preprocessing layer to the front of VGG
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False
    

    
    # useful for getting number of classes
    folders = glob(f'train/face/{name}/*')
    

    # our layers - you can add more if you want
    x = Flatten()(vgg.output)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    #x = Dropout(0.5)(x)
    prediction = Dense(len(folders), activation='softmax')(x)

    # create a model object
    model = Model(inputs=vgg.input, outputs=prediction)

    # view the structure of the model
    model.summary()

    # tell the model what cost and optimization method to use
    model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr = 0.0001),
    metrics=['accuracy']
    )


    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                    target_size = (224, 224),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(valid_path,
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical')

    #r=model.fit_generator(training_set,
    #                        samples_per_epoch = 8000,
    #                        nb_epoch = 5,
    #                        validation_data = test_set,
    #                        nb_val_samples = 2000)

    # Enter the number of training and validation samples here
    nb_train_samples = 100
    nb_validation_samples = 50
    batch_size = 16

    # fit the model
    r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch= len(training_set),
    validation_steps= len(test_set))


    # loss
    #plt.plot(r.history['loss'], label='train loss')
    #plt.plot(r.history['val_loss'], label='val loss')
    #plt.legend()
    #plt.show()
    #plt.savefig('LossVal_loss')

    # accuracies
    #plt.plot(r.history['acc'], label='train acc')
    #plt.plot(r.history['val_acc'], label='val acc')
    #plt.legend()
    #plt.show()
    #plt.savefig('AccVal_acc')

    import tensorflow as tf

    from keras.models import load_model

    model.save('facefeatures_new_model.h5')

def make_model(name):
    '''
    Creates a model of {name}'s face using the VGG16 and the images datasets of {name}
    '''
    
    # final image size
    IMAGE_SIZE = [224, 224]
    
    # link to directories of datasets
    training_dir = 'train/face/'
    test_dir = 'test/face/'
    
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) # instantiates vgg and adds a preprocessing layer to front
    
    # disables training on pre-existing layers
    for layer in vgg.layers:
        layer.trainable = False
        
    train_classes = glob(f'{training_dir}{name}/*') # gets all available classes from dataset
    
    # layers
    x = Flatten()(vgg.output) # flattens data into a 1D array
    # x = Dense(1000, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1000, activation='relu')(x)
    pred = Dense(len(train_classes), activation='softmax')(x) 
    
    model = Model(inputs=vgg.input, outputs=pred)
    
    # prints summary of model structure
    model.summary()
    
    # tells our model what optimization and cost method to use
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # generates tensor data batches
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # gets all images from datasets
    training_set = train_datagen.flow_from_directory(training_dir,
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                     class_mode='categorical')
    test_set = test_datagen.flow_from_directory(test_dir,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='categorical')
    
    r = model.fit(training_set,
                  validation_data=test_set,
                  epochs=5,
                  steps_per_epoch=len(training_set),
                  validation_steps=len(test_set))
    
    # loss
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.savefig('LossVal_loss')

    # accuracies
    plt.plot(r.history['acc'], label='train acc')
    plt.plot(r.history['val_acc'], label='val acc')
    plt.legend()
    plt.show()
    plt.savefig('AccVal_acc')
    
    model.save('facefeatures_new_model.h5')
    
if __name__ == '__main__':
        
    cam = cv2.VideoCapture(0)
    make_data(cam, 'vincent')
    make_model2('vincent')