# Import all packages needed
import pandas as pd
import numpy as np
import math
import cv2
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
from keras.constraints import maxnorm

DATA_PATH = './Data3'
# Load the training data and split them into a smaller validation set with test size of 80% and the remaining 20% for valid
driving_log = []
def loadData(data_path):
    with open('./Data3/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            driving_log.append(line)
    # Use Train_test_split from sklearn package to split data set in training and validation set    
    d_train, d_valid = train_test_split(driving_log, test_size=0.2, random_state=42)
    # Print out the number of data in the sets
    print(len(d_train),len(d_valid))
    print(len(driving_log)) 
    return d_train, d_valid

# The generator is used to load all images from one time stamp into the pipeline. It also corrects the steering angle from the side cameras
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Import also the images on the side
                for i in range(3):
                    name = './Data3/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    measurement = float(batch_sample[3])
                    # i == 1 second row in data package - will be corrected with +0.25
                    if i==1:
                        measurement = measurement + 0.25
                    # i == 2 third row in data package - will be corrected with -0.25
                    elif i==2:
                        measurement = measurement - 0.25
                    
                    # Append all images in pipeline
                    images.append(image)
                    measurements.append(measurement)

            augmented_images, augmented_measurements = [], []
            # Flip the images to double the data number
            for image, measurement in zip(images, measurements):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_measurements.append(measurement * -1.0)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            yield shuffle(X_train, y_train)
            
def resize(x):
    """
    Resizes the images in the supplied tensor to the original dimensions of the NVIDIA model (66x200)
    """
    from keras.backend import tf as ktf
    return ktf.image.resize_images(x, [66, 200])
            
def build_model():
    """
    NVIDIA CNN model
    with given input shape and resize of the images
    """
    INPUT_SHAPE = (160,320,3)
    model = Sequential()
    # model.add(Lambda(lambda x: x[:,80:,:,:], input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,24), (60,60)), input_shape=INPUT_SHAPE))
    # Normalization
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    model.add(Lambda(resize))
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    # model.add(Dense(1164,activation='relu',W_constraint=maxnorm(3))) # Added
    # model.add(Dropout(0.3)) # Added
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.3)) # Added
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.3)) # Added
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model

def train_model(model, train_generator, validation_generator, d_train, d_valid, batch_size):
    
    model.compile(loss='mse', optimizer='adam')
    
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(d_train)*6,
                    validation_data=validation_generator,nb_val_samples=len(d_valid)*6, 
                    nb_epoch=5, verbose=1)
    # save the model
    model.save('model46.h5')
    return history_object
    
def main():
    # This is the main pipeline with full training processing
    # Set our batch size
    batch_size=4
    
    d_train, d_valid = loadData(DATA_PATH)    
    train_generator = generator(d_train, batch_size=batch_size)
    validation_generator = generator(d_valid, batch_size=batch_size)
    model = build_model()
    train_samples = len(d_train)
    print(len(d_train))
    print(len(d_valid))
    validation_samples = len(d_valid)
    history_object = train_model(model, train_generator, validation_generator, d_train, d_valid, batch_size)
    
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
  
if __name__ == "__main__": main()