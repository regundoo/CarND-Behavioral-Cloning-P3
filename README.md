# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behaviour
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteUp_Images/nvidia_cnn_architecture.png "Model Visualization"
[image2]: ./WriteUp_Images/ModelBash.PNG "model visu"
[image3]: ./WriteUp_Images/DataSet3Epoch4.PNG "traiing Save"
[image4]: ./WriteUp_Images/steeringAngles.PNG "Recovery Image"
[image5]: ./WriteUp_Images/cameraAngle.PNG "Recovery Image"
[image6]: ./WriteUp_Images/center.jpg "center"
[image7]: ./WriteUp_Images/left.jpg "left Image"
[image8]: ./WriteUp_Images/right.jpg "right Image"
[video1]: ./video.mp4 "video"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Different training sets are used to variate the input of the model. Therefore, different input files are used in the code.

### Model Architecture and Training Strategy

#### 1. Datasets

The model used in this exercise uses data from the provided simulator. We ended up with 3 different datasets for training:
1. Udacity's dataset on track 1 
2. Manually created dataset on track 1
3. Manually created dataset where the car was driven the other way around the track and some "recovery" modes where performed 

All collected data sets are combined to one big data package and one csv file with the measurements.

#### 2. Dataset Exploration

The measurements of the steering angles are very inconsistent, since the keyboard is used as an input. The model acts very inconsistent to such kind of data. As also seen in the image below, we have a huge amount of measurements with a "0" degree angle. Therefore, the model is overtrained to drive straight.

![alt text][image4]

#### 3. Dataset Split

From the collected data set, a validation set is split up to validate the trained model. The "train_test_split" function is used for that purpose. The set is split into parts of 80% training data and 20% validation data.

```
d_train, d_valid = train_test_split(driving_log, test_size=0.2, random_state=42)
```
This gives us 10510 data points for training and 2628 data points for validation. The number of points in the two sets will be increased later with image processing.

#### 4. Camera and Steering Wheel Angle Calibration

As known from the image capturing, the cameras on the side of the car collect a different point of few from the scene. Therefore, the steering angle for those measurements must be adapted.

![alt text][image5]

Centre Image  
![alt text][image6]  

Left Image  
![alt text][image7]

Right Image  
![alt text][image8]

First, we add a correction value to images captured by either left or right cameras:
* for the left camera we want the car to steer to the right (positive offset)
* for the right camera we want the car to steer to the left (negative offset)

```
 if i==1:
    measurement = measurement + 0.25
 elif i==2:
    measurement = measurement - 0.25
```

#### 5. Image Horizontal Flip

To increase the amount of training and validation data without driving many times around the track, we use an image flip on every defined training image. This gives us 6 times the data we had before. We are flipping the images from the centre camera and the images from the side cameras. This leads to a total amount of 63060 data points for training.


#### 6. Keras Image Generator

We are generating new and augmented images on the fly as we train the model. This gives us some advantages in the storage for the model. We are not loading all the data at once in the memory. The mentioned image flip happens in the data generator.

```
def generator(samples, batch_size=32):
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
```

## Model

The initially used model is a NVIDIA research model with few modifications. The structure of the model is the following:

![alt text][image1]

### Model Tweaks

However, the model is slightly addopted.
* A crop of the image is added. Since we know that we do not need the top of the images for training. This could confuse the model
* The size of the images is reshaped to 66x200
* The Images are normalized

## Model Architecture

The full architecture of the model is as follows:

![alt text][image2]

* After the Cropping layer, two lambda layers are added to normalize the images and to reshape them to the mentioned values.
* The next steps are 5 Conv2D layers with different output shapes
* After flattening the model, the last 4 layers are for Dense the model. 
* Output node has the shape 1

All in all, 252,219 parameters are trained for this model.

It was also tried to use some Dropout functions, but this was not very effective. The batch generation helps to prevent overfitting 

## Activations and Regularization

For all layers, the activation function is the ReLu function. ELU activation was also tried but ReLu gives the best results for training. 

## Training and Results

The model was trained using Adam for optimisation. After lots of training and tweaking of parameters, a model was created which drives the first track without any problems. The video can be found here:

![video][video1]

The accuracy loss is the following:

![alt text][image3]



# Conclusion

There are several different ways to perform better results from training:
* create better input data. Creating data without the keyboard creates much smoother data. This will lead to better in put values
* use data from the second track. Additional data can be used to create even more data from the second track. Since the lighting is changing in this scene, image pre-processing becomes very important. A random shadow can be added to the images to prevent the model to get scared of such lighting changes.
* Speed and throttle can be implemented into the trained model. This will lead to a faster and smoother driving speed.
